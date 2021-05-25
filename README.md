# Retail_Transactions
Use case on Transactions dataset


Description – The dataset contains ~2 years of data starting from 12/1/2009 till 11/9/2011. The task is to predict for all customers in this window who will come back to buy any product next month (11/9/2011 – 12/9/2011). 
The variable names and their descriptions are as follows:

1.	InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
2.	StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
3.	Description: Product (item) name. Nominal.
4.	Quantity: The quantities of each product (item) per transaction. Numeric.
5.	InvoiceDate: Invoice Date and time. Numeric, the day and time when each transaction was generated.
6.	UnitPrice: Unit price. Numeric, Product price per unit in sterling.
7.	CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
8.	Country: Country name. Nominal, the name of the country where each customer resides.

Business problem :
Predict whether a customer is going to buy a product next month.
