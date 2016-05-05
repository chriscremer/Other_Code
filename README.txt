
Miscellaneous Code


CSC2508 - Database Course - University of Toronto
Argo is an automated mapping layer that runs on top of a traditional RDBMS, and presents the JSON data model directly 
to the application/user. Argo has been previously shown to have a significant speedup over document-oriented databases,
specifically MongoDB. This was shown by evaluating the systems using the benchmark NoBench. This benchmark evaluates 
queries on relatively simple JSON documents. Therefore, in order to gain a more comprehensive assessment of Argo, I 
extended the NoBench data to include more complex data, such as highly nested objects and arrays with more elements. 
Here I show that increasing the complexity of the data reduces the speedup that Argo has over MongoDB.
