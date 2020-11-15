# Netflix
Each week the CFPB sends thousands of consumers’ complaints about financial products and services to companies for response. Those complaints are published here after the company responds or after 15 days, whichever comes first. By adding their voice, consumers help improve the financial marketplace. The dataset was collected from [Kaggle](https://www.kaggle.com/cfpb/us-consumer-finance-complaints).


# Learning Outcomes
Following my analysis of my [Netflix Viewing Habits](https://github.com/kjonina/personal_Netflix/) and [analysing Netflix Content](https://github.com/kjonina/Netflix), I decided to continue my quest to analyse data and update README.md on datasets that I have not yet finished.
I have a lot of holes in my knowledge so I decided to outline some learning outcomes for myself:

The purpose of this analysis for myself is to: 
- [ ] deal with parts of code I struggle with: loc, iloc splitting text data, creating linegraph and horizontal barcharts, barcharts by a category. 
- [ ] run analysis on ['narrative']
- [ ] create a predictive model for complaints 
- [ ] continue fancy-schmancy code to update README.md such as links to external url and links, etc. 


### Variables

There are 18 variables and 556 000 observations.

['date_received', 'product', 'sub_product', 'issue', 'sub_issue','consumer_complaint_narrative', 'company_public_response', 'company','state', 'zipcode', 'tags', 'consumer_consent_provided', 'submitted_via', 'date_sent_to_company', 'company_response_to_consumer','timely_response', 'consumer_disputed?', 'complaint_id']

New Variables:
['days_to_process']  compares the number of days between ['date_sent_to_company'] and ['date_received']

