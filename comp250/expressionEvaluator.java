//Christopher Cremer
//260407933

import java.io.*;
import java.util.*;
import java.lang.*;

public class expressionEvaluator {
    static String delimiters="+-*%()";

    
    /* This method evaluates the given arithmetic expression and returns
     * its Integer value. The method throws an Exception if the expression
     * is malformed.*/
    static Integer evaluate( String expr ) throws Exception {
       
        StringTokenizer st = new StringTokenizer( expr , delimiters , true );    
        
        Stack<Integer> number = new Stack<Integer>();
        Stack<String> operator = new Stack<String>();
        int factor =0;
        while (st.hasMoreTokens())
        {
        	String token = st.nextToken();
        	if (token.charAt(0) >= 48 && token.charAt(0) <= 57)
        	{
        		number.push(new Integer(token));
        		factor++;
        	}
        	if ((token.charAt(0) >= 40 && token.charAt(0) <=43) || token.charAt(0) == 37 || token.charAt(0) == 45)
        	{
        		operator.push(token);
        		if (token.charAt(0) == 40)
        		{
        			factor= 0;
        		}
        		if (token.charAt(0) ==41)
        		{
        			operator.pop();
        			operator.pop();
        			if(operator.isEmpty())
        			{
        				factor=1;
        			}
        			else if (operator.peek().equals("+") || operator.peek().equals("-") || operator.peek().equals("*") || operator.peek().equals("%"))
        				{
        					factor=2;
        				}
        		}
        	}
        	if (factor == 2)
        	{
        		if (operator.peek().equals("+"))
        		{
        			number.push(number.pop()+number.pop());
        			operator.pop();
        		}
        		else if (operator.peek().equals("-"))
        		{
        			Integer temp = number.pop();
        			number.push(number.pop()-temp);
        			operator.pop();
        		}
        		else if (operator.peek().equals("*"))
        		{
        			number.push(number.pop()*number.pop());
        			operator.pop();
        		}
        		else if (operator.peek().equals("%"))
        		{
        			Integer temp = number.pop();
        			number.push(number.pop()/temp);
        			operator.pop();
        		}
        		factor =1;
        	}
        }

       
       if (!operator.isEmpty())
       {
    	   throw new Exception();
       }
       
       return number.pop();
       
    } 


    /* This method repeatedly asks the user for an expression and evaluates it.
       The loop stops when the user enters an empty expression */
    public void queryAndEvaluate() throws Exception {    
        String line;
        BufferedReader stdin = new BufferedReader(new InputStreamReader( System.in ) );
         System.out.println("Enter an expression");
        line = stdin.readLine();    
    
        while ( line.length() > 0 ) {
            try {
                Integer value = evaluate( line );
                System.out.println("value = " + value );
            }
            catch (Exception e)
            {
                // write something here!
            	 System.out.println("Malformed expression");
            }
            System.out.println( "Enter an expression" );
            line = stdin.readLine();    
        } // end of while loop
    } // end of query and evaluate
        
    public static void main(String args[]) throws Exception {
         expressionEvaluator e=new expressionEvaluator();
         e.queryAndEvaluate();
     } // end of main
}

// 2013
