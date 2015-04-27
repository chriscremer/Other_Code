import java.lang.Math.*;

class expressionTreeNode {
    private Object value;
    private expressionTreeNode leftChild, rightChild, parent;
    
    expressionTreeNode() {
        value = null; 
        leftChild = rightChild = parent = null;
    }
    
    // Constructor
    /* Arguments: String s: Value to be stored in the node
                  expressionTreeNode l, r, p: the left child, right child, and parent of the node to created      
       Returns: the newly created expressionTreeNode               
    */
    expressionTreeNode(String s, expressionTreeNode l, expressionTreeNode r, expressionTreeNode p) {
        value = s; 
        leftChild = l; 
        rightChild = r;
        parent = p;
    }
    
    /* Basic access methods */
    Object getValue() { return value; }

    expressionTreeNode getLeftChild() { return leftChild; }

    expressionTreeNode getRightChild() { return rightChild; }

    expressionTreeNode getParent() { return parent; }


    /* Basic setting methods */ 
    void setValue(Object o) { value = o; }
    
    // sets the left child of this node to n
    void setLeftChild(expressionTreeNode n) { 
        leftChild = n; 
        n.parent = this; 
    }
    
    // sets the right child of this node to n
    void setRightChild(expressionTreeNode n) { 
        rightChild = n; 
        n.parent=this; 
    }
    

    // Returns the root of the tree describing the expression s
    // Watch out: it makes no validity checks whatsoever!
    expressionTreeNode(String s) {
        // check if s contains parentheses. If it doesn't, then it's a leaf
        if (s.indexOf("(")==-1) setValue(s);
        else {  // it's not a leaf

            /* break the string into three parts: the operator, the left operand,
               and the right operand. ***/
            setValue( s.substring( 0 , s.indexOf( "(" ) ) );
            // delimit the left operand 2008
            int left = s.indexOf("(")+1;
            int i = left;
            int parCount = 0;
            // find the comma separating the two operands
            while (parCount>=0 && !(s.charAt(i)==',' && parCount==0)) {
                if ( s.charAt(i) == '(' ) parCount++;
                if ( s.charAt(i) == ')' ) parCount--;
                i++;
            }
            int mid=i;
            if (parCount<0) mid--;

        // recursively build the left subtree
            setLeftChild(new expressionTreeNode(s.substring(left,mid)));
    
            if (parCount==0) {
                // it is a binary operator
                // find the end of the second operand.07
                while ( ! (s.charAt(i) == ')' && parCount == 0 ) )  {
                    if ( s.charAt(i) == '(' ) parCount++;
                    if ( s.charAt(i) == ')' ) parCount--;
                    i++;
                }
                int right=i;
                setRightChild( new expressionTreeNode( s.substring( mid + 1, right)));
        }
    }
    }


    // Returns a copy of the subtree rooted at this node... 2013
    expressionTreeNode deepCopy() {
        expressionTreeNode n = new expressionTreeNode();
        n.setValue( getValue() );
        if ( getLeftChild()!=null ) n.setLeftChild( getLeftChild().deepCopy() );
        if ( getRightChild()!=null ) n.setRightChild( getRightChild().deepCopy() );
        return n;
    }
    
    // Returns a String describing the subtree rooted at a certain node.
    public String toString() {
        String ret = (String) value;
        if ( getLeftChild() == null ) return ret;
        else ret = ret + "(" + getLeftChild().toString();
        if ( getRightChild() == null ) return ret + ")";
        else ret = ret + "," + getRightChild().toString();
        ret = ret + ")";
        return ret;
    } 


    // Returns the value of the the expression rooted at a given node
    // when x has a certain value
    double evaluate(double x) 
    {
    	double ret =0; //the result of the evaluation

    	if (getValue().equals("x"))	//if variable, return variable
    	{
    		return x;
    	}
    	if (getLeftChild() == null)	//if number, return number, because only numbers/variable have null child but if variable it returns before getting to this code
    	{
    		return Double.parseDouble((String)getValue());
    	}
    	switch ((String)getValue())	//evaluate the left and right child depending on the operator
    	{
    	case "add":
    		ret = getLeftChild().evaluate(x) + getRightChild().evaluate(x);
    		break;
    	case "minus":
    		ret = getRightChild().evaluate(x) - getLeftChild().evaluate(x);
    		break;
    	case "mult":
    		ret = getLeftChild().evaluate(x) * getRightChild().evaluate(x);
    		break;
    	case "sin":
    		ret = Math.sin(getLeftChild().evaluate(x));
    		break;
    	case "cos":
    		ret = Math.cos(getLeftChild().evaluate(x));
    		break;
    	case "exp":
    		ret = Math.exp(getLeftChild().evaluate(x));
    		break;
    	default:
    		break;
    	}
    	
    return ret;	
    }                         

    /* returns the root of a new expression tree representing the derivative of the
    expression represented by the tree rooted at the node on which it is called ***/
    expressionTreeNode differentiate() 
    {
   
    	expressionTreeNode n = new expressionTreeNode();
    	
    	if (getValue().equals("x"))	//base case, derivative of x
    	{
    		n.setValue("1");
    	}
    	else if (getLeftChild() == null)	//base case, derivative of constant
    	{
    		n.setValue("0");
    	}
    	switch ((String)getValue())	//derivatives of the operators
    	{
    	case "add":
    		n.setValue(getValue());
    		n.setLeftChild(getLeftChild().differentiate());
    		n.setRightChild(getRightChild().differentiate());
    		break;
    	case "minus":
    		n.setValue(getValue());
    		n.setLeftChild(getLeftChild().differentiate());
    		n.setRightChild(getRightChild().differentiate());
    		break;
    	case "mult":
    		n.setValue("add");
    		n.setLeftChild(new expressionTreeNode("mult", getLeftChild().differentiate(), getRightChild(), this));
    		n.setRightChild(new expressionTreeNode("mult", getLeftChild(), getRightChild().differentiate(), this));
    		break;
    	case "sin":
    		n.setValue("mult");
    		n.setLeftChild(new expressionTreeNode("cos", getLeftChild(), null, this));
    		n.setRightChild(getLeftChild().differentiate());
    		break;
    	case "cos":
    		n.setValue("mult");
    		n.setLeftChild(new expressionTreeNode("-sin", getLeftChild(), null, this));
    		n.setRightChild(getLeftChild().differentiate());
    		break;
    	case "exp":
    		n.setValue("mult");
    		n.setLeftChild(new expressionTreeNode("exp", getLeftChild(), null, this));
    		n.setRightChild(getLeftChild().differentiate());
    		break;
    	default:
    		break;
    	}
    	
    	return n; //return the differentiated node tree
    }
    
    /* Extra-credit */
    expressionTreeNode simplify() {
    // WRITE YOUR CODE HERE

    return null; // remove this
    }

    
    public static void main(String args[]) {
        //expressionTreeNode e = new expressionTreeNode("mult(add(2,x),cos(x))");
        expressionTreeNode e = new expressionTreeNode("cos(minus(x,4))");
    	//expressionTreeNode e = new expressionTreeNode("cos(add(mult(3.1416,x),exp(sin(minus(x,1)))))");
    	//expressionTreeNode e = new expressionTreeNode("add(x,add(x,1))");
    	//expressionTreeNode e = new expressionTreeNode("mult(x,1)");
    	//expressionTreeNode e = new expressionTreeNode("exp(x)");
        System.out.println("expression: " +e);
        System.out.println("evaluate: " +e.evaluate(1));
        System.out.println("differentiate: " +e.differentiate());
        
        
       //System.out.println("\n" + e.toString());
        //System.out.println("\n" + e.evaluate(1));
        
    }
}
