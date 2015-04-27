import java.util.*;
import java.io.*;
import java.lang.*;

// This class implements a google-like search engine
public class searchEngine {

    public HashMap wordIndex;                  // this will contain a set of pairs (String, LinkedList of Strings)        
    public directedGraph internet;             // this is our internet graph
    public static double DAMPING_FACTOR = 0.5;        // this value is suggested by the authors of Google   
        
        
    // Constructor initializes everything to empty data structures
    // It also sets the location of the internet files. 2013
    searchEngine() {
        // Below is the directory that contains all the internet files
        htmlParsing.internetFilesLocation = "internetFiles";
        wordIndex = new HashMap();                
        internet = new directedGraph();                                
    } // end of constructor
        
        
    // Returns a String description of a searchEngine
    public String toString () {
        return "wordIndex:\n" + wordIndex + "\ninternet:\n" + internet;
    }


    // This does a graph traversal of the internet, starting at the given url.
    // For each new vertex seen, it updates the wordIndex, the internet graph,
    // and the set of visited vertices.
        
    void traverseInternet(String url) throws Exception {
        /* WRITE SOME CODE HERE */
    	Queue<String> q = new LinkedList<String>();		//queue for breadth first search
    	
    	internet.addVertex(url);	//add the start vertex of the graph
    	internet.setVisited(url, true);	//set it to visited
    	q.add(url);					//add it to the queue
    	while(!q.isEmpty())
    	{
    		String website = q.remove();	//make variable 'website' the first url in the queue
    		LinkedList content = htmlParsing.getContent(website);	//linkedlist of the content of the page
    		Iterator j = content.iterator();//iterator on the content
    		while(j.hasNext())//until the iterator is done
    		{
    			String word = (String)j.next();//go through each word
    			if (wordIndex.containsKey(word))//if the haspmap already contains the keyword
    			{
    				LinkedList addToList = new LinkedList();//linkedlist to add new website
    				addToList = (LinkedList)wordIndex.get(word);//point the new list to the word in the hashmap
    				if (!addToList.contains(website))//if the same word comes up more than once in the website
    				{
    					addToList.add(website);//add the website to the list
    				}
    			}
    			else
    			{
    				LinkedList createList = new LinkedList();//list for new word
    				createList.add(website);//add the website to the list
    				wordIndex.put(word, createList);//put the list in the hashmap
    			}
    			
    		}
    		
    		LinkedList<String> links = htmlParsing.getLinks(website);//list of the links on the page
    		Iterator i = links.iterator();//iterator for the links
    		while(i.hasNext())
    		{
    			String neighbor = (String)i.next();//iterate through the neighbors
    			
    			if(!internet.getVisited(neighbor))//if not visited
    			{
    				internet.addVertex(neighbor);//add vertex to graph
    				internet.addEdge(website, neighbor);//add edge
    				internet.setVisited(neighbor, true);//set to visited
    				q.add(neighbor);//add to queue
    			}
    			else
    			{
    				internet.addEdge(website, neighbor);//if already visited just add edge
    			}
    		}	
    		System.out.println(website);//to see order of visits
    	}
    	
    	
    	
    	
    	
        /* Hints
           0) This should take about 50-70 lines of code (or less)
           1) To parse the content of the url, call
           htmlParsing.getContent(url), which returns a LinkedList of Strings 
           containing all the words at the given url. Also call htmlParsing.getLinks(url).
           and assign their results to a LinkedList of Strings.
           2) To iterate over all elements of a LinkedList, use an Iterator,
           like described in the text of the assignment
           3) Refer to the description of the LinkedList methods at
           http://java.sun.com/j2se/1.4.2/docs/api 
           You will most likely need to use the methods contains(String s), 
           addLast(String s), iterator()
           4) Refer to the description of the HashMap methods at
           http://java.sun.com/j2se/1.4.2/docs/api 
           You will most likely need to use the methods containsKey(String s), 
           get(String s), put(String s, LinkedList l).  
        */

        
        
    } // end of traverseInternet


    /* This computes the pageRanks for every vertex in the internet graph.
       It will only be called after the internet graph has been constructed using 
       traverseInternet.
       Use the iterative procedure described in the text of the assignment to
       compute the pageRanks for every vertices in the graph. 
                      
       This method will probably fit in about 30 lines.
    */
    void computePageRanks() {
        /* WRITE YOUR CODE HERE */
                
    	LinkedList allVertices = internet.getVertices();//list of all the urls in the graph
    	Iterator i = allVertices.iterator();//iterator for all the whole graph
    	while(i.hasNext())//0th iteration
		{
			String vertex = (String)i.next();
			internet.setPageRank(vertex, 1);//make all the ranks equal to 1
		}
  
    	for (int j=0; j<100; j++)//100 repetitions
    	{
    		Iterator k = allVertices.iterator();//iterator for all vertices again
    		while(k.hasNext())
    		{
    			String vertex = (String)k.next();
    			LinkedList edgesInto = internet.getEdgesInto(vertex);//list of the urls that link to the page
    			Iterator h = edgesInto.iterator();//iterator for the urls that link to the page
    			double sum=0;//sum for the ranks/out-degree of the sites that link to the page
    			while(h.hasNext())
    			{
    				String vertexInto = (String)h.next();
    				sum = sum + (internet.getPageRank(vertexInto)/internet.getOutDegree(vertexInto));//PR/out-degree
    			}
    			 
    			internet.setPageRank(vertex, (1-DAMPING_FACTOR)+(DAMPING_FACTOR)*(sum));//page rank equation
    		}
    		
    		
    	}
    	
    } // end of computePageRanks
                
        
    /* Returns the URL of the page with the high page-rank containing the query word
       Returns the String "" if no web site contains the query.
       This method can only be called after the computePageRanks method has been executed.
       Start by obtaining the list of URLs containing the query word. Then return the URL 
       with the highest pageRank.
       This method should take about 25 lines of code.
    */
    String getBestURL(String query) {
        /* WRITE YOUR CODE HERE */
    	
    	if (!wordIndex.containsKey(query))// if the query is not in the index
    	{
    		return "";
    	}
    	else
    	{
    		LinkedList websiteIndex = (LinkedList)wordIndex.get(query);//list of the websites for that query
    		Iterator i = websiteIndex.iterator();//iterator for the websites of that query
    		double highestRank=0;//variable for saving highest rank
    		String bestWebsite = null;//varaible for rankest ranked website
    		while(i.hasNext())
    		{
    			String website = (String)i.next();
    			if (internet.getPageRank(website) > highestRank)// if the rank is higher
    			{
    				highestRank = internet.getPageRank(website);//set new highest rank
    				bestWebsite = website;//set new highest ranked website
    			}
    		}
    		return bestWebsite;//return the highest ranked website
    	}
    	
  
    	
    } // end of getBestURL
        

        

    // You shouldn't need to modify the main method, except maybe for debugging purposes
    public static void main(String args[]) throws Exception{                

        // create an object of type searchEngine
        searchEngine google = new searchEngine();

        // to debug your program, start with.
        //google.traverseInternet("http://www.cs.mcgill.ca");
        google.traverseInternet("http://www.cs.mcgill.ca/~blanchem/250/a.html");

        // When your program is working on the small example, move on to
        // google.traverseInternet("http://www.cs.mcgill.ca/index.html");

        // this is just for debugging purposes
        System.out.println(google);

        google.computePageRanks();

        // this is just for debugging purposes
        System.out.println(google);
                
        BufferedReader stndin = new BufferedReader( new InputStreamReader(System.in) );
        String query;
        do {
            System.out.print( "Enter query: " );
            query = stndin.readLine();
            if ( query != null && query.length() > 0 ) {
                System.out.println("Best site = " + google.getBestURL(query));
            }
        } while ( query != null && query.length() > 0 );                                
    } // end of main
}
