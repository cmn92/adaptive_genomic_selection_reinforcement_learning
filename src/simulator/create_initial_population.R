#########################################################################
# create_initial_population.R
#
# Purpose:
# Create the initial maize candidate population from the simulated inbred
# founder lines.
#
# The breeding structure is:
# - 100 inbred founder lines
# - 100 random biparental crosses
# - 1 F1 individual per cross
# - 10 doubled haploid (DH) individuals per F1
# - 1000 candidate DH individuals in total
#########################################################################

##########################################################################
# 1. Load libraries and working directory
##########################################################################
library(AlphaSimR)
setwd("~/adaptive_genomic_selection_reinforcement_learning")
##########################################################################
# 2. set random seed for reproducibility
##########################################################################
set.seed(12345)


##########################################################################
# 3. Define input and output files
##########################################################################
founderPop_file <- "data/founder_population.RData"

output_file <- "data/initial_candidate_population.RData"

###########################################################################
# 4. Check if the founder population file exists
###########################################################################
if (!file.exists(founderPop_file)) {
  stop(paste("Founder population file", founderPop_file, "does not exist. Please run the founder population simulation first."))
}

###########################################################################
# 5. Load the founder population and simulation parameters
###########################################################################
load(founderPop_file)

if (!exists("founderPop") || !exists("SP")) {
  stop("The founder population or simulation parameters are not found in the loaded file.")
}

###########################################################################
# 6. Convert founder haplotypes into an AlphaSimR population
###########################################################################
founder_lines <- newPop(
    rawPop = founderPop,
    simParam = SP
)

###########################################################################
# 7. Define the crossing design
###########################################################################

num_crosses <- 100
f1_per_cross <- 1
dh_per_f1 <- 10

############################################################################
# 8. Create random biparental crosses from the founder lines
############################################################################
f1_population <- randCross(
    pop = founder_lines,
    nCrosses = num_crosses,
    nProgeny = f1_per_cross,
    balance = TRUE,
    simParam = SP
)

###########################################################################
# 9. Generate doubled haploid (DH) individuals from the F1 population
###########################################################################
candidate_population <- makeDH(
    pop = f1_population,
    nDH = dh_per_f1,
    simParam = SP
)

###########################################################################
# 10. Assign candidate identifiers to the DH individuals
###########################################################################
candidate_population@id <- sprintf(
    "CAND_%04d",
    seq_len(candidate_population@nInd)
)

###########################################################################
# 11. Save the populations and simulation parameters 
###########################################################################
save(
    founderPop,
    f1_population,
    candidate_population,
    SP,
    file = output_file
)

###########################################################################
# 12. Print confirmation message
###########################################################################
cat("\nInitial maize candidate population created successfully.\n")

cat("Number of founder lines:", founder_lines@nInd, "\n")

cat("Number of crosses:", num_crosses, "\n")

cat("Number of F1 individuals:", f1_population@nInd, "\n")

cat("Number of DH individuals per F1:", dh_per_f1, "\n")

cat("Number of candidate DH individuals:", candidate_population@nInd, "\n")

cat("Saved to file:", output_file, "\n")
