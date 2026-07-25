#########################################################################
#create_founders.R
#Purpose:
# Create a maize-like founder population for the adaptive GS simulation.

#The founder haplotypes are simulated using MacS (Chen et al., 2009) through
# AlphaSimR package (Gaynor et al., 2020).
# The simulated genome containes:
# - 100 inbred founder lines 
# - 10 chromosomes 
# - 300 segregating sites per chromosome 
# - 10 additive QTL per chromosome 
# - 200 SNP markers per chromosome 
# - one additive quantitative trait 
# - broad-sense heritability of 0.40
#########################################################################

#########################################################################
# 1. Load libraries and source files
#########################################################################
library(AlphaSimR)

#########################################################################
# 2. set random seed for reproducibility
#########################################################################
set.seed(12345)

#########################################################################
# 3. Define parameters for the founder population
#########################################################################
num_founders <- 100
num_Chr <- 10
segSites_per_Chr <- 300
inbred_founders <- TRUE
species <- "MAIZE"

#################################################################################
# 4. Define trait and marker parameters
#################################################################################
qtl_per_Chr <- 10
snp_markers_per_Chr <- 200

trait_mean <- 0
trait_variance <- 1
trait_heritability <- 0.4

##########################################################################
# 5. Create the founder population
##########################################################################

founderPop <- runMacs(nInd = num_founders,
                     nChr = num_Chr,
                     segSites = segSites_per_Chr,
                     inbred = inbred_founders,
                     species = species)

###########################################################################
# 6. Create the AlphaSimR simulation parameters
###########################################################################
SP = SimParam$new(founderPop)

##########################################################################
# 7. Define an additive quantitative trait
##########################################################################
SP$addTraitA(nQtlPerChr = qtl_per_Chr,
                mean = trait_mean,
                var = trait_variance)

###########################################################################
# 8. Define SNP markers for genotyping
###########################################################################
SP$addSnpChip(nSnpPerChr = snp_markers_per_Chr)

###########################################################################
# 9. Set the broad-sense heritability for the trait
###########################################################################
SP$setVarE(h2 = trait_heritability)

###########################################################################
# 10. Save the founder population to a file
###########################################################################
save(founderPop,SP, file = "~/adaptive_genomic_selection_reinforcement_learning/data/founder_population.RData")
print("Founder population created and saved to file."  )

######################################################################### 
# 12. Print basic confirmation information 
######################################################################### 
cat("\nMaize-like founder population created successfully.\n") 
cat("Number of founders:", num_founders, "\n") 
cat("Number of chromosomes:", num_Chr, "\n") 
cat("Segregating sites per chromosome:", segSites_per_Chr, "\n") 
cat("QTL per chromosome:", qtl_per_Chr, "\n") 
cat("SNP markers per chromosome:", snp_markers_per_Chr, "\n") 
cat("Trait heritability:", trait_heritability, "\n") 