start_date="20250104"
#### Always one day more than you want
end_date="20250930"

# Loop through the date range
current_date="$start_date"

while [ "$current_date" != "$end_date" ]; do
    year=${current_date:0:4}
    month=${current_date:4:2}
    day=${current_date:6:2}
    
    if [[ "$day" == "04" ]]; then
        # Run process_date.sh for the first of the month
        first_of_month="${current_date:0:6}01"
        ./run_train.sh "$first_of_month" 
        cd Production
        ./run_fcst.sh "$current_date"
        cd ..
    else
        # Run not_first_of_month.sh for all other days
        cd Production
        ./run_fcst.sh "$current_date"
        cd ..
    fi

    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done