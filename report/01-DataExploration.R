library(readxl)
library(dplyr)

# File is in the main repo folder
file_path <- "data/who_air_quality_2024.xlsx"

excel_sheets(file_path)

air_data <- read_excel(file_path, sheet = "Update 2024 (V6.1)")

head(air_data)
summary(air_data)


