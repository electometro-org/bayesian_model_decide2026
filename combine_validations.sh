#!/bin/bash

find ./results/validations/ -name "age*.csv" -exec cat {} + > ./results/validations/age_kombiniert.csv

find ./results/validations/ -name "education*.csv" -exec cat {} + > ./results/validations/eduaction_kombiniert.csv

find ./results/validations/ -name "region*.csv" -exec cat {} + > ./results/validations/region_kombiniert.csv

find ./results/validations/ -name "gender*.csv" -exec cat {} + > ./results/validations/gender_kombiniert.csv
