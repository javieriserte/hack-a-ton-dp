jq -r '.[] | [">", (.id | tostring) , "\n", (.sequence | tostring) ] | join("")' disorder_test.json > disorder_test.fasta
jq -r '.[] | [">", (.id | tostring) , "\n", (.sequence | tostring) ] | join("")' disorder_train.json > disorder_train.fasta
