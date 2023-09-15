

if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("SummarizedExperiment")
BiocManager::install("DREAM4")

pkgs = c("xlsx","readxl","writexl");
new = pkgs[!(pkgs%in%installed.packages()[,"Package"])]
if (length(new)){for(pkg in new) install.packages(pkg, dependencies = TRUE);}
sapply(pkgs, require, character.only = TRUE);
