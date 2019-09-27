#!/bin/bash
timestamp=$(`date`)
kaggle competitions submit -c criteo-display-ad-challenge -f predicted_test.csv.gz -m "submitted at ${timestamp}"
