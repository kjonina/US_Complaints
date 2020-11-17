# US_Complaints
Each week the CFPB sends thousands of consumers’ complaints about financial products and services to companies for response. Those complaints are published here after the company responds or after 15 days, whichever comes first. By adding their voice, consumers help improve the financial marketplace. The dataset was collected from [Kaggle](https://www.kaggle.com/cfpb/us-consumer-finance-complaints).


# Learning Outcomes
Following my analysis of my [Netflix Viewing Habits](https://github.com/kjonina/personal_Netflix/) and [analysing Netflix Content](https://github.com/kjonina/Netflix), I decided to continue my quest to analyse data and update README.md on datasets that I have not yet finished.
I have a lot of holes in my knowledge so I decided to outline some learning outcomes for myself:

The purpose of this analysis for myself is to: 
- [ ] deal with parts of code I struggle with: loc, iloc splitting text data, creating linegraph and horizontal barcharts, barcharts by a category. 
- [ ] run text analysis on ['narrative']
- [ ] create a predictive model for complaints

### Variables

There are 18 variables and 556 000 observations.

['date_received', 'product', 'sub_product', 'issue', 'sub_issue','consumer_complaint_narrative', 'company_public_response', 'company','state', 'zipcode', 'tags', 'consumer_consent_provided', 'submitted_via', 'date_sent_to_company', 'company_response_to_consumer','timely_response', 'consumer_disputed?', 'complaint_id']

New Variables:
['days_to_process']  compares the number of days between ['date_sent_to_company'] and ['date_received']

## Exploring EDA

### What states have the most number of complaints?

![state](https://github.com/kjonina/US_Complaints/blob/changes/Graph/state.png)

### What products have the most number of complaints?

![products](https://github.com/kjonina/US_Complaints/blob/changes/Graph/Products.png)

### What products have the most number of complaints?

![products](https://github.com/kjonina/US_Complaints/blob/changes/Graph/Products.png)

### What was the most popular method of submission??

![submission](https://github.com/kjonina/US_Complaints/blob/changes/Graph/submission.png)


There is a big discrepancy between methods of submission. How was fax more popular than email? 

| Submission Method | Count |
| ---------------- | ------------- |
| Web | 361338 |
| Referral | 109379 |
| Phone | 40026 |
| Postal mail | 36752 |
| Fax| 8118 |
| Email| 344 |


### What was the most common response from the companies?

![response](https://github.com/kjonina/US_Complaints/blob/changes/Graph/response.png)

| Submission Method | Count |
| ---------------- | ------------- |
| Closed with explanation | 404293 |
| Closed with non-monetary relief | 70237 |
| Closed with monetary relief | 38262 |
| Closed without relief | 17909 |
| Closed | 13399 |
| Closed with relief | 5305 |
| In progress | 3763 |
| Untimely response | 2789 |


### Was that the companies response any good?

![timely](https://github.com/kjonina/US_Complaints/blob/changes/Graph/timely.png)

Only 2.5% of compaints did not receive a timely response.

| Was there a timely response? | Count |
| ---------------- | ------------- |
| Yes | 541909 |
| No | 14048 |


### What is the length of time between the date sent to the company and date received by product?

![response_by_product](https://github.com/kjonina/US_Complaints/blob/changes/Graph/response_by_product.png)

## how are credit card, mortgage etc, negative? Check this out for future
 
 # TO DO -> 
 
 - draw a time graph for date_received + date_sent_to_company!
 - draw a graph the company by product type
 - draw a graph for days_process
 - find the  most common words in Narrative. 

 - creating a wordcloud for issue 
 - find methods to analyse text.

