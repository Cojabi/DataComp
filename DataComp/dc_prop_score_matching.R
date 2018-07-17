#### Propensity scoring matching for feature comparism between ANM and adni
# https://imai.princeton.edu/research/files/matchit.pdf#subsubsection.4.1.3

library(MatchIt)
library(optmatch)

# vector storing the diagnosises, can be used to pop_match all of them in a loop
diags = c("") #, "ad_", "mci_", "ctl_")

#######################  FUNCTIONS  ################################
prop_score_matching = function(formula, data_path, save_path, jitter_plot=TRUE){
  
  data = read.csv(data_path)
  # set rownames to operate with patient ID's
  rownames(data) = data$X
  
  # nearest neighbor matching using the caliper option
  # caliper: number of standard deviations that the distance measures of neighbors are allowed to be apart
  m_near = matchit(formula, data=data, method="nearest", caliper=0.5)
  
  if (jitter_plot){
    plot(m_near, type="jitter")
  }
  
  # get match matrix
  mats = m_near$match.matrix
  colnames(mats) = c("Match")
  
  # save
  write.csv(mats, save_path)
  print(m_near)
}

###################  RUN FOR ALL DIAGNOSISES ######################
for(diag in diags){
  # create save and data path based on diagnosis
  save_path = paste("/home/colin/SCAI/git/Dataset_comparison/compare_sites_data/", paste(diag, "matches.csv", sep=""), sep="")
  data_path = paste("/home/colin/SCAI/git/Dataset_comparison/compare_sites_data/", paste(diag, "prop_compare_data.csv", sep = ""), sep="")

  prop_score_matching(data_path, save_path)
}


################### ADNI vs ANM FreeSurfer6 Edition ######################
data_path = "/home/colin/SCAI/git/Dataset_comparison/compare_sites_data/MRI_F6/prop_compare_data_F6.csv"
save_path = "/home/colin/SCAI/git/Dataset_comparison/compare_sites_data/MRI_F6/matches_F6.csv"
formula = Cohort ~ PTGENDER+PTEDUCAT+AGE+APOE4+Diagnosis

prop_score_matching(formula, data_path, save_path)
