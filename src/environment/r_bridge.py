"""
r_bridge.py

Python interface to the AlphaSimR breeding simulator.

The bridge:
1. Starts an embedded R session through rpy2.
2. Sources the existing R simulator files.
3. Loads the initial AlphaSimR candidate population.
4. Accepts Python-selected candidate indices.
5. Supports both:
   - a complete one-call breeding cycle; and
   - staged within-generation phenotyping for active learning and RL.
6. Returns metrics and candidate information to Python.

Index convention
----------------
Python uses zero-based indices:
    0, 1, ..., n - 1

R uses one-based indices:
    1, 2, ..., n

The bridge performs this conversion internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.vectors import IntVector, ListVector
except ImportError as exc:
    raise ImportError(
        "rpy2 is required to use the R breeding simulator. "
        "Install it with:\n"
        "python -m pip install rpy2"
    ) from exc


class RBridgeError(RuntimeError):
    """Raised when the R breeding simulator cannot complete an operation."""


class RBreedingBridge:
    """
    Persistent bridge between Python and the AlphaSimR simulator.

    One instance represents one breeding-program episode. The current
    population is held in the embedded R session and updated after each
    completed breeding generation.
    """

    REQUIRED_R_FILES = (
        "phenotyping.R",
        "genomic_prediction.R",
        "prediction_uncertainty.R",
        "parent_selection.R",
        "next_generation.R",
        "breeding_cycle.R",
    )

    def __init__(
        self,
        project_root: str | Path,
        population_file: str | Path = (
            "data/initial_candidate_population.RData"
        ),
        seed: int = 12345,
    ) -> None:
        """
        Initialize the bridge and load the R simulator.

        Parameters
        ----------
        project_root:
            Root directory of the project repository.

        population_file:
            Saved initial candidate population. A relative path is
            interpreted relative to project_root.

        seed:
            Base random seed for reproducibility.
        """
        self.project_root = Path(project_root).expanduser().resolve()
        self.simulator_dir = self.project_root / "src" / "simulator"

        population_path = Path(population_file).expanduser()
        if not population_path.is_absolute():
            population_path = self.project_root / population_path

        self.population_file = population_path.resolve()
        self.base_seed = self._validate_positive_integer(seed, "seed")

        self.generation = 1

        self._current_population: Any | None = None
        self._initial_population: Any | None = None
        self._sim_param: Any | None = None
        self._last_cycle_result: Any | None = None

        # Temporary state used by staged active-learning/RL interaction.
        self._training_population: Any | None = None
        self._predicted_population: Any | None = None
        self._rrblup_model: Any | None = None
        self._current_prediction_result: Any | None = None

        self._phenotyped_python_indices = np.empty(
            0,
            dtype=np.int64,
        )
        self._phenotype_tables: list[pd.DataFrame] = []

        self._validate_paths()
        self._initialize_r_session()
        self.reset(seed=self.base_seed)

    @staticmethod
    def _finite_or_zero(value: Any) -> float:
        """Return a finite float, using zero for undefined numeric results."""
        numeric = float(value)

        if not np.isfinite(numeric):
            return 0.0

        return numeric

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        """Check that all required project files exist."""
        if not self.project_root.is_dir():
            raise FileNotFoundError(
                f"Project root does not exist: {self.project_root}"
            )

        if not self.population_file.is_file():
            raise FileNotFoundError(
                "Initial population file was not found:\n"
                f"{self.population_file}"
            )

        missing_files = [
            self.simulator_dir / filename
            for filename in self.REQUIRED_R_FILES
            if not (self.simulator_dir / filename).is_file()
        ]

        if missing_files:
            formatted = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                "The following R simulator files are missing:\n"
                f"{formatted}"
            )

    def _initialize_r_session(self) -> None:
        """Load AlphaSimR and source all required simulator functions."""
        try:
            ro.r(
                'suppressPackageStartupMessages(library("AlphaSimR"))'
            )

            for filename in self.REQUIRED_R_FILES:
                script_path = self.simulator_dir / filename
                ro.r["source"](str(script_path))

            required_functions = (
                "run_breeding_cycle",
                "phenotype_selected",
                "fit_rrblup_prediction",
                "select_top_parents",
                "create_next_generation",
                "compute_prediction_uncertainty",
                "select_highest_pev_candidates",
            )

            missing_functions = [
                name
                for name in required_functions
                if name not in ro.globalenv
            ]

            if missing_functions:
                raise RBridgeError(
                    "The following R functions were not loaded: "
                    + ", ".join(missing_functions)
                )

        except RBridgeError:
            raise
        except Exception as exc:
            raise RBridgeError(
                "Failed to initialize the R simulator. Confirm that R, "
                "AlphaSimR, and rpy2 are installed correctly."
            ) from exc

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """
        Reset the breeding program to the initial candidate population.

        Returns
        -------
        dict
            Basic information about the reset population.
        """
        reset_seed = self.base_seed if seed is None else seed
        reset_seed = self._validate_positive_integer(
            reset_seed,
            "seed",
        )

        try:
            loaded_names = [
                str(name)
                for name in ro.r["load"](str(self.population_file))
            ]

            if "candidate_population" not in loaded_names:
                raise RBridgeError(
                    "'candidate_population' is missing from the RData file."
                )

            if "SP" not in loaded_names:
                raise RBridgeError(
                    "'SP' is missing from the RData file."
                )

            self._initial_population = ro.globalenv[
                "candidate_population"
            ]
            self._current_population = self._initial_population
            self._sim_param = ro.globalenv["SP"]

            ro.r["set.seed"](reset_seed)

            self.base_seed = reset_seed
            if "bridge_generation" in loaded_names:
                self.generation = int(
                    ro.globalenv["bridge_generation"][0]
                )
            else:
                self.generation = 1
            self._last_cycle_result = None
            self._clear_generation_state()

            return {
                "generation": self.generation,
                "population_size": self.population_size,
                "individual_ids": self.get_candidate_ids(),
            }

        except RBridgeError:
            raise
        except Exception as exc:
            raise RBridgeError(
                "Failed to reset the breeding simulator from "
                f"{self.population_file}."
            ) from exc

    def save_program_state(
        self,
        path: str | Path,
    ) -> Path:
        """Save the current population, SimParam, and generation number."""
        self._require_population()

        state_path = Path(path).expanduser().resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ro.globalenv["candidate_population"] = (
                self._current_population
            )
            ro.globalenv["SP"] = self._sim_param
            ro.globalenv["bridge_generation"] = IntVector(
                [int(self.generation)]
            )
            ro.r["save"](
                "candidate_population",
                "SP",
                "bridge_generation",
                file=str(state_path),
            )
        except Exception as exc:
            raise RBridgeError(
                "Failed to save the current breeding-program state."
            ) from exc

        return state_path

    def load_program_state(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """Restore a state saved by save_program_state()."""
        state_path = Path(path).expanduser().resolve()

        if not state_path.is_file():
            raise FileNotFoundError(
                f"Saved breeding-program state not found: {state_path}"
            )

        try:
            loaded_names = [
                str(name)
                for name in ro.r["load"](str(state_path))
            ]

            if "candidate_population" not in loaded_names:
                raise RBridgeError(
                    "Saved state is missing 'candidate_population'."
                )

            if "SP" not in loaded_names:
                raise RBridgeError("Saved state is missing 'SP'.")

            self._current_population = ro.globalenv[
                "candidate_population"
            ]
            self._initial_population = self._current_population
            self._sim_param = ro.globalenv["SP"]

            if "bridge_generation" in loaded_names:
                self.generation = int(
                    ro.globalenv["bridge_generation"][0]
                )
            else:
                self.generation = 1

            self._last_cycle_result = None
            self._clear_generation_state()

            return {
                "generation": self.generation,
                "population_size": self.population_size,
                "individual_ids": self.get_candidate_ids(),
            }

        except RBridgeError:
            raise
        except Exception as exc:
            raise RBridgeError(
                "Failed to load a saved breeding-program state."
            ) from exc

    def set_trait_heritability(
        self,
        heritability: float,
    ) -> None:
        """Update AlphaSimR environmental variance using broad-sense h2."""
        numeric = float(heritability)

        if not np.isfinite(numeric):
            raise ValueError("'heritability' must be finite.")

        if not 0.0 < numeric <= 1.0:
            raise ValueError(
                "'heritability' must be greater than zero and at most one."
            )

        try:
            ro.globalenv["SP"] = self._sim_param
            ro.globalenv["bridge_h2"] = ro.FloatVector([numeric])
            ro.r("SP$setVarE(h2 = bridge_h2)")
            self._sim_param = ro.globalenv["SP"]
        except Exception as exc:
            raise RBridgeError(
                "Failed to update the simulator heritability."
            ) from exc

    # ------------------------------------------------------------------
    # Population access
    # ------------------------------------------------------------------

    @property
    def population_size(self) -> int:
        """Return the number of candidates in the current population."""
        self._require_population()

        try:
            n_ind = ro.r["slot"](
                self._current_population,
                "nInd",
            )
            return int(n_ind[0])
        except Exception as exc:
            raise RBridgeError(
                "Could not determine the current population size."
            ) from exc

    def get_candidate_ids(self) -> list[str]:
        """Return current candidate identifiers in population order."""
        self._require_population()

        try:
            ids = ro.r["slot"](
                self._current_population,
                "id",
            )
            return [str(value) for value in ids]
        except Exception as exc:
            raise RBridgeError(
                "Could not extract candidate identifiers."
            ) from exc

    def get_marker_matrix(self, snp_chip: int = 1) -> np.ndarray:
        """
        Extract the current candidate-by-marker SNP matrix.

        Marker values are generally coded as dosages 0, 1, and 2.
        """
        self._require_population()

        snp_chip = self._validate_positive_integer(
            snp_chip,
            "snp_chip",
        )

        try:
            marker_matrix_r = ro.r["pullSnpGeno"](
                self._current_population,
                snpChip=snp_chip,
                simParam=self._sim_param,
            )

            with localconverter(
                default_converter + numpy2ri.converter
            ):
                marker_matrix = (
                    ro.conversion.get_conversion().rpy2py(
                        marker_matrix_r
                    )
                )

            marker_matrix = np.asarray(
                marker_matrix,
                dtype=np.float64,
            )

            if marker_matrix.ndim != 2:
                raise RBridgeError(
                    "The extracted marker data are not two-dimensional."
                )

            if marker_matrix.shape[0] != self.population_size:
                raise RBridgeError(
                    "Marker matrix row count does not match "
                    "the population size."
                )

            return marker_matrix

        except RBridgeError:
            raise
        except Exception as exc:
            raise RBridgeError(
                "Could not extract the SNP marker matrix from AlphaSimR."
            ) from exc

    def get_candidate_data(
        self,
        include_markers: bool = False,
    ) -> dict[str, Any]:
        """
        Return information that a Python selection strategy can use.
        """
        if not isinstance(include_markers, bool):
            raise TypeError("'include_markers' must be Boolean.")

        data: dict[str, Any] = {
            "generation": self.generation,
            "population_size": self.population_size,
            "individual_ids": self.get_candidate_ids(),
        }

        if include_markers:
            data["marker_matrix"] = self.get_marker_matrix()

        return data

    # ------------------------------------------------------------------
    # Staged within-generation workflow
    # ------------------------------------------------------------------

    def start_generation(self) -> dict[str, Any]:
        """
        Clear temporary within-generation state without changing population.

        Call this before a staged active-learning or RL phenotyping
        sequence.
        """
        self._require_population()
        self._clear_generation_state()

        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "individual_ids": self.get_candidate_ids(),
            "number_phenotyped": 0,
        }

    @property
    def number_currently_phenotyped(self) -> int:
        """Return the number phenotyped in the current staged generation."""
        return int(self._phenotyped_python_indices.size)

    def get_phenotyped_indices(self) -> np.ndarray:
        """Return zero-based indices phenotyped in this generation."""
        return self._phenotyped_python_indices.copy()

    def get_unphenotyped_indices(self) -> np.ndarray:
        """Return zero-based indices not yet phenotyped this generation."""
        all_indices = np.arange(
            self.population_size,
            dtype=np.int64,
        )

        if self._phenotyped_python_indices.size == 0:
            return all_indices

        mask = np.ones(
            self.population_size,
            dtype=bool,
        )
        mask[self._phenotyped_python_indices] = False
        return all_indices[mask]

    def phenotype_batch(
        self,
        selected_indices: Sequence[int] | np.ndarray,
        *,
        reps: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Phenotype an additional candidate batch.

        Multiple calls accumulate a combined training population for the
        current generation. The same candidate cannot be phenotyped twice.
        """
        self._require_population()

        selected_python = self._validate_selected_indices(
            selected_indices
        )
        reps = self._validate_positive_integer(reps, "reps")

        overlap = np.intersect1d(
            selected_python,
            self._phenotyped_python_indices,
        )

        if overlap.size > 0:
            raise ValueError(
                "The following candidates were already phenotyped in "
                f"this generation: {overlap[:10].tolist()}"
            )

        batch_seed = (
            self.base_seed
            + self.generation
            + self.number_currently_phenotyped
            if seed is None
            else self._validate_positive_integer(seed, "seed")
        )

        selected_r = IntVector(
            (selected_python + 1).astype(np.int32).tolist()
        )

        try:
            ro.r["set.seed"](batch_seed)

            phenotype_function = ro.globalenv[
                "phenotype_selected"
            ]

            phenotype_result = phenotype_function(
                population=self._current_population,
                selected_indices=selected_r,
                reps=reps,
                simParam=self._sim_param,
            )

            new_training_batch = phenotype_result.rx2(
                "phenotyped_population"
            )

            new_phenotype_table = self._r_dataframe_to_pandas(
                phenotype_result.rx2("phenotype_table")
            )

            if self._training_population is None:
                self._training_population = new_training_batch
            else:
                self._training_population = ro.r["mergePops"](
                    ListVector(
                        [
                            self._training_population,
                            new_training_batch,
                        ]
                    )
                )

            self._phenotyped_python_indices = np.concatenate(
                [
                    self._phenotyped_python_indices,
                    selected_python,
                ]
            )

            new_phenotype_table = new_phenotype_table.copy()
            new_phenotype_table["phenotyping_batch"] = (
                len(self._phenotype_tables) + 1
            )
            self._phenotype_tables.append(new_phenotype_table)

            # Existing predictions are stale after adding new phenotypes.
            self._rrblup_model = None
            self._predicted_population = None
            self._current_prediction_result = None

            return {
                "generation": self.generation,
                "batch_indices": selected_python.copy(),
                "batch_size": int(selected_python.size),
                "total_phenotyped": (
                    self.number_currently_phenotyped
                ),
                "phenotype_table": new_phenotype_table,
            }

        except Exception as exc:
            raise RBridgeError(
                "R failed while phenotyping a candidate batch."
            ) from exc

    def get_combined_phenotype_table(self) -> pd.DataFrame:
        """Return all phenotype records collected this generation."""
        if not self._phenotype_tables:
            return pd.DataFrame()

        return pd.concat(
            self._phenotype_tables,
            ignore_index=True,
        )

    def fit_current_model(
        self,
        *,
        trait: int = 1,
    ) -> dict[str, Any]:
        """
        Fit RR-BLUP using all phenotypes collected this generation.

        Predictions are generated for the entire current candidate
        population.
        """
        self._require_population()

        if self._training_population is None:
            raise RBridgeError(
                "No candidates have been phenotyped. "
                "Call phenotype_batch() before fitting RR-BLUP."
            )

        trait = self._validate_positive_integer(trait, "trait")

        try:
            prediction_function = ro.globalenv[
                "fit_rrblup_prediction"
            ]

            prediction_result = prediction_function(
                training_population=self._training_population,
                candidate_population=self._current_population,
                trait=trait,
                simParam=self._sim_param,
            )

            self._current_prediction_result = prediction_result
            self._rrblup_model = prediction_result.rx2("model")
            self._predicted_population = prediction_result.rx2(
                "predicted_population"
            )

            prediction_table = self._r_dataframe_to_pandas(
                prediction_result.rx2("prediction_table")
            )

            prediction_accuracy = self._finite_or_zero(
                prediction_result.rx2(
                    "prediction_accuracy"
                )[0]
            )

            return {
                "generation": self.generation,
                "training_population_size": (
                    self.number_currently_phenotyped
                ),
                "prediction_accuracy": prediction_accuracy,
                "prediction_table": prediction_table,
            }

        except Exception as exc:
            raise RBridgeError(
                "R failed while fitting the current RR-BLUP model."
            ) from exc
        
    def compute_current_uncertainty(
        self,
        *,
        trait: int = 1,
        snp_chip: int = 1,
        n_cores: int = 1,
    ) -> dict[str, Any]:
        """
        Calculate candidate-level prediction error variance using G-BLUP.

        All phenotype batches collected in the current generation are used as
        the training population. The returned table contains uncertainty for
        both phenotyped and unphenotyped candidates.

        Parameters
        ----------
        trait:
            One-based AlphaSimR trait index.

        snp_chip:
            One-based SNP-chip index.

        n_cores:
            Number of processor cores used by rrBLUP::A.mat().

        Returns
        -------
        dict
            Complete uncertainty table, ranked unphenotyped table and
            estimated variance components.
        """
        self._require_population()

        if self._training_population is None:
            raise RBridgeError(
                "No candidates have been phenotyped. "
                "Call phenotype_batch() before computing uncertainty."
            )

        trait = self._validate_positive_integer(trait, "trait")
        snp_chip = self._validate_positive_integer(
            snp_chip,
            "snp_chip",
        )
        n_cores = self._validate_positive_integer(
            n_cores,
            "n_cores",
        )

        try:
            uncertainty_function = ro.globalenv[
                "compute_prediction_uncertainty"
            ]

            uncertainty_result = uncertainty_function(
                candidate_population=self._current_population,
                training_population=self._training_population,
                trait=trait,
                snp_chip=snp_chip,
                n_cores=n_cores,
                simParam=self._sim_param,
            )

            uncertainty_table = self._r_dataframe_to_pandas(
                uncertainty_result.rx2("uncertainty_table")
            )

            unphenotyped_uncertainty = (
                self._r_dataframe_to_pandas(
                    uncertainty_result.rx2(
                        "unphenotyped_uncertainty"
                    )
                )
            )

            genetic_variance = float(
                uncertainty_result.rx2(
                    "genetic_variance"
                )[0]
            )

            residual_variance = float(
                uncertainty_result.rx2(
                    "residual_variance"
                )[0]
            )

            return {
                "generation": self.generation,
                "training_population_size": (
                    self.number_currently_phenotyped
                ),
                "uncertainty_table": uncertainty_table,
                "unphenotyped_uncertainty": (
                    unphenotyped_uncertainty
                ),
                "genetic_variance": genetic_variance,
                "residual_variance": residual_variance,
            }

        except Exception as exc:
            raise RBridgeError(
                "R failed while calculating prediction uncertainty."
            ) from exc

    def get_current_prediction_table(self) -> pd.DataFrame:
        """Return predictions from the most recently fitted model."""
        if self._current_prediction_result is None:
            raise RBridgeError(
                "No current prediction model is available. "
                "Call fit_current_model() first."
            )

        return self._r_dataframe_to_pandas(
            self._current_prediction_result.rx2(
                "prediction_table"
            )
        )

    def finalize_generation(
        self,
        *,
        number_of_parents: int = 20,
        number_of_crosses: int = 100,
        f1_per_cross: int = 1,
        dh_per_f1: int = 10,
        trait: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Select parents and create the next candidate population.

        A fitted model must exist. If a new phenotype batch was added
        after fitting, fit_current_model() must be called again first.
        """
        self._require_population()

        if self._predicted_population is None:
            raise RBridgeError(
                "No fitted prediction model is available. "
                "Call fit_current_model() first."
            )

        number_of_parents = self._validate_positive_integer(
            number_of_parents,
            "number_of_parents",
        )
        number_of_crosses = self._validate_positive_integer(
            number_of_crosses,
            "number_of_crosses",
        )
        f1_per_cross = self._validate_positive_integer(
            f1_per_cross,
            "f1_per_cross",
        )
        dh_per_f1 = self._validate_positive_integer(
            dh_per_f1,
            "dh_per_f1",
        )
        trait = self._validate_positive_integer(trait, "trait")

        cycle_seed = (
            self.base_seed + self.generation - 1
            if seed is None
            else self._validate_positive_integer(seed, "seed")
        )

        try:
            current_gv = np.asarray(
                ro.r["gv"](self._current_population),
                dtype=np.float64,
            )[:, trait - 1]

            population_mean_before = float(
                np.mean(current_gv)
            )
            population_variance_before = float(
                np.var(current_gv, ddof=1)
            )

            selection_result = ro.globalenv[
                "select_top_parents"
            ](
                predicted_population=self._predicted_population,
                number_of_parents=number_of_parents,
                trait=trait,
            )

            selected_parents = selection_result.rx2(
                "selected_parents"
            )

            selection_table = self._r_dataframe_to_pandas(
                selection_result.rx2("selection_table")
            )

            selected_parent_mean_gv = float(
                selection_table[
                    "true_breeding_value"
                ].mean()
            )

            next_generation_result = ro.globalenv[
                "create_next_generation"
            ](
                selected_parents=selected_parents,
                number_of_crosses=number_of_crosses,
                f1_per_cross=f1_per_cross,
                dh_per_f1=dh_per_f1,
                generation_number=self.generation + 1,
                seed=cycle_seed,
                simParam=self._sim_param,
            )

            next_population = next_generation_result.rx2(
                "candidate_population"
            )

            next_gv = np.asarray(
                ro.r["gv"](next_population),
                dtype=np.float64,
            )[:, trait - 1]

            prediction_table = self.get_current_prediction_table()

            prediction_accuracy = self._finite_or_zero(
                self._current_prediction_result.rx2(
                    "prediction_accuracy"
                )[0]
            )

            cycle_summary = pd.DataFrame(
                {
                    "generation": [self.generation],
                    "population_size": [self.population_size],
                    "number_phenotyped": [
                        self.number_currently_phenotyped
                    ],
                    "phenotyping_fraction": [
                        self.number_currently_phenotyped
                        / self.population_size
                    ],
                    "replications": [1],
                    "phenotyping_cost_units": [
                        self.number_currently_phenotyped
                    ],
                    "training_population_size": [
                        self.number_currently_phenotyped
                    ],
                    "prediction_accuracy": [
                        prediction_accuracy
                    ],
                    "number_of_selected_parents": [
                        number_of_parents
                    ],
                    "population_mean_gv_before": [
                        population_mean_before
                    ],
                    "population_variance_gv_before": [
                        population_variance_before
                    ],
                    "selected_parent_mean_gv": [
                        selected_parent_mean_gv
                    ],
                    "selection_differential": [
                        selected_parent_mean_gv
                        - population_mean_before
                    ],
                    "next_generation_mean_gv": [
                        float(np.mean(next_gv))
                    ],
                    "next_generation_variance_gv": [
                        float(np.var(next_gv, ddof=1))
                    ],
                    "realized_genetic_gain": [
                        float(np.mean(next_gv))
                        - population_mean_before
                    ],
                }
            )

            completed_generation = self.generation
            phenotyped_indices = self.get_phenotyped_indices()
            phenotype_table = (
                self.get_combined_phenotype_table()
            )

            self._current_population = next_population
            self.generation += 1

            result = {
                "generation": completed_generation,
                "phenotyped_indices": phenotyped_indices,
                "phenotype_table": phenotype_table,
                "prediction_table": prediction_table,
                "selection_table": selection_table,
                "cycle_summary": cycle_summary,
                "next_population_size": self.population_size,
                "next_individual_ids": self.get_candidate_ids(),
            }

            self._clear_generation_state()
            return result

        except RBridgeError:
            raise
        except Exception as exc:
            raise RBridgeError(
                "R failed while finalizing the breeding generation."
            ) from exc

    # ------------------------------------------------------------------
    # Complete one-call breeding cycle
    # ------------------------------------------------------------------

    def step(
        self,
        selected_indices: Sequence[int] | np.ndarray,
        *,
        number_of_parents: int = 20,
        number_of_crosses: int = 100,
        f1_per_cross: int = 1,
        dh_per_f1: int = 10,
        reps: int = 1,
        trait: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Run one complete breeding cycle in a single call.

        This remains the main interface for random, fixed, and diversity
        baselines. Active learning and RL can use the staged methods.
        """
        self._require_population()
        self._clear_generation_state()

        selected_python = self._validate_selected_indices(
            selected_indices
        )

        number_of_parents = self._validate_positive_integer(
            number_of_parents,
            "number_of_parents",
        )
        number_of_crosses = self._validate_positive_integer(
            number_of_crosses,
            "number_of_crosses",
        )
        f1_per_cross = self._validate_positive_integer(
            f1_per_cross,
            "f1_per_cross",
        )
        dh_per_f1 = self._validate_positive_integer(
            dh_per_f1,
            "dh_per_f1",
        )
        reps = self._validate_positive_integer(reps, "reps")
        trait = self._validate_positive_integer(trait, "trait")

        cycle_seed = (
            self.base_seed + self.generation - 1
            if seed is None
            else self._validate_positive_integer(seed, "seed")
        )

        selected_r = IntVector(
            (selected_python + 1).astype(np.int32).tolist()
        )

        try:
            cycle_result = ro.globalenv[
                "run_breeding_cycle"
            ](
                candidate_population=self._current_population,
                selected_indices=selected_r,
                generation_number=self.generation,
                number_of_parents=number_of_parents,
                number_of_crosses=number_of_crosses,
                f1_per_cross=f1_per_cross,
                dh_per_f1=dh_per_f1,
                reps=reps,
                trait=trait,
                seed=cycle_seed,
                simParam=self._sim_param,
            )

            cycle_summary = self._r_dataframe_to_pandas(
                cycle_result.rx2("cycle_summary")
            )
            selection_table = self._r_dataframe_to_pandas(
                cycle_result.rx2("selection_table")
            )
            phenotype_table = self._r_dataframe_to_pandas(
                cycle_result.rx2("phenotype_table")
            )
            prediction_table = self._r_dataframe_to_pandas(
                cycle_result.rx2("prediction_table")
            )

            self._last_cycle_result = cycle_result
            self._current_population = cycle_result.rx2(
                "next_candidate_population"
            )

            completed_generation = self.generation
            self.generation += 1
            self._clear_generation_state()

            return {
                "generation": completed_generation,
                "selected_indices": selected_python.copy(),
                "cycle_summary": cycle_summary,
                "phenotype_table": phenotype_table,
                "prediction_table": prediction_table,
                "selection_table": selection_table,
                "next_population_size": self.population_size,
                "next_individual_ids": self.get_candidate_ids(),
            }

        except Exception as exc:
            raise RBridgeError(
                "R failed while running breeding generation "
                f"{self.generation}."
            ) from exc

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _r_dataframe_to_pandas(
        r_dataframe: Any,
    ) -> pd.DataFrame:
        """Convert an R data.frame to an independent pandas DataFrame."""
        try:
            with localconverter(
                default_converter + pandas2ri.converter
            ):
                result = (
                    ro.conversion.get_conversion().rpy2py(
                        r_dataframe
                    )
                )

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)

            return result.reset_index(drop=True).copy()

        except Exception as exc:
            raise RBridgeError(
                "Failed to convert an R data.frame to pandas."
            ) from exc

    def _validate_selected_indices(
        self,
        selected_indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Validate zero-based candidate indices supplied by Python."""
        indices = np.asarray(selected_indices)

        if indices.ndim != 1:
            raise ValueError(
                "'selected_indices' must be one-dimensional."
            )

        if indices.size == 0:
            raise ValueError("'selected_indices' cannot be empty.")

        if not np.issubdtype(indices.dtype, np.number):
            raise TypeError(
                "'selected_indices' must contain numbers."
            )

        numeric_indices = indices.astype(float)

        if np.any(~np.isfinite(numeric_indices)):
            raise ValueError(
                "'selected_indices' cannot contain NaN or infinity."
            )

        if np.any(numeric_indices % 1 != 0):
            raise ValueError(
                "'selected_indices' must contain whole numbers."
            )

        indices = indices.astype(np.int64)

        if np.any(indices < 0):
            raise IndexError(
                "Python candidate indices must be non-negative."
            )

        if np.any(indices >= self.population_size):
            raise IndexError(
                "At least one selected index exceeds the current "
                f"population size of {self.population_size}."
            )

        if np.unique(indices).size != indices.size:
            raise ValueError(
                "'selected_indices' cannot contain duplicates."
            )

        return indices

    @staticmethod
    def _validate_positive_integer(
        value: Any,
        name: str,
    ) -> int:
        """Validate a scalar positive integer."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"'{name}' must be a positive integer."
            )

        if not np.isscalar(value):
            raise TypeError(
                f"'{name}' must be a scalar positive integer."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"'{name}' must be a positive integer."
            ) from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"'{name}' must be finite.")

        if numeric_value % 1 != 0 or numeric_value < 1:
            raise ValueError(
                f"'{name}' must be a positive integer."
            )

        return int(numeric_value)

    def _require_population(self) -> None:
        """Ensure that a current AlphaSimR population is loaded."""
        if self._current_population is None:
            raise RBridgeError(
                "No current population is loaded. Call reset() first."
            )

    def _clear_generation_state(self) -> None:
        """Clear temporary state for a staged breeding generation."""
        self._training_population = None
        self._predicted_population = None
        self._rrblup_model = None
        self._current_prediction_result = None

        self._phenotyped_python_indices = np.empty(
            0,
            dtype=np.int64,
        )
        self._phenotype_tables = []
