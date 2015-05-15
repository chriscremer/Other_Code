//Christopher Cremer
//260407933


public class PQHeapAbstract {
	private static int[] heap;
	public static int currentPosition;
	
	public PQHeapAbstract(int N)
	{
		heap = new int[N+1];
		currentPosition = 0;
	}
	
	
	private void heapifyUp(int index)
	{
		if (index > 1)
		{
			int parentIndex = (index)/2;
			if (heap[index] < heap[parentIndex])
			{
				int temp = heap[index];
				heap[index] = heap[parentIndex];
				heap[parentIndex] = temp;
				heapifyUp(parentIndex);
			}
		}
	}
	
	
	private void heapifyDown(int index)
	{
		int n = currentPosition;
		int smallerChild;
		if (2*index > n || currentPosition == 1)
		{
			return;
		}
		else if (2*index < n)
		{
			int left = 2*index;
			int right = (2*index)+1; 
			if (heap[left] <= heap[right])
			{
				smallerChild = left;
			}
			else
			{
				smallerChild = right;
			}	
		}
		else
		{
			smallerChild = 2*index;
		}
		if (heap[smallerChild] < heap[index])
		{
			int temp = heap[smallerChild];
			heap[smallerChild] = heap[index];
			heap[index] = temp;
		}
		heapifyDown(smallerChild);
	}
	
	
	
	public void Insert(int v)
	{
		currentPosition++;
		heap[currentPosition] = v;
		heapifyUp(currentPosition);
	}
	
	
	
	public int FindMin()
	{
		return heap[1];
	}
	
	
	
	public void Delete(int index)
	{
		if (index >= currentPosition)
		{
			currentPosition--;
			return;
		}
		
			heap[index] = heap[currentPosition];
			currentPosition--;
			heapifyDown(index);
		
	}
	
	
	
	public void ExtractMin()
	{
		Delete(1);
	}
	
	
	public int getSize()
	{
		return currentPosition;
	}
	
	public String toString()
	{
		if (currentPosition == 0)
		{
			return "Heap is empty";
		}
		String heapString = "";
		for (int i=1; i<=currentPosition; i++)
		{
			heapString = heapString + " " + heap[i]; 
		}
		return heapString;
	}

}
