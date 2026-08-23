A DMPK group is investigating a promising new lead, called CPD7, which is converted to metabolites M1 and M2 within the bloodstream. The group has collected PK and PD data from two distinct patient cohorts. Investigate the data and determine which of the measured species' exposure best predicts the pharmacological response. Data are available in the following files:

/workspace/data/pk_concentrations.csv
Columns: subject_id, cohort, time_hr, analyte, conc_ng_ml
Three analytes present: parent, M1, M2

/workspace/data/pd_response.csv
Columns: subject_id, cohort, response

/workspace/data/subjects.csv
Columns: subject_id, cohort, dose_mg, n_doses, tau_hr

Save your result to /workspace/output/result.json using this schema:

{
  "nominated_species": "string",
  "single_dose_association": 0.0,
  "multi_dose_association": 0.0
}
