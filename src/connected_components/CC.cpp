/*********************************************************************************
This piece of program contained can be used, copied, modified, merged, published, 
and/or have copies distributed for academic or research purposes only without 
restriction under the following conditions:

1. The above header and this permission notice shall be included in all copies or 
substantial portions of the program

2. The software is provided "as is", without warranty of any kind, express or implied, 
including but not limited to the warranties of merchantability, fitness for a particular 
purpose and non-infringement. In no event shall the author(s) be liable for any claim, 
damages or liability, whether in an action of contract, tort or otherwise, arising from, 
out of or in connection with this program.

3. If you use this piece of code for research purposes, refer to 

Tola, Engin. 2006 June 12. Homepage. <http://cvlab.epfl.ch/~tola/index.htm>

4. An acknowledgement note should be included as: 

     "The software used here was originally created by Tola, Engin. 2006 June 12. 
	 Homepage. <http://cvlab.epfl.ch/~tola/index.htm>"

**************************************************************************************/

// ConnectedComponentLabeler.cpp: implementation of the CConnectedComponentLabeler class.
//
//////////////////////////////////////////////////////////////////////


#include <connected_components/CC.h> // ConnectedComponents
#include <iostream>
#include <string>
#include <cstdlib>

namespace ConnectedComponents {

using std::cout;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CC::CC()
{
	m_MaskArray = NULL;
	m_height = 0;
	m_width  = 0;
	m_nAreaThreshold = 0;
	m_ObjectNumber = 0;
}

CC::~CC()
{
	m_Components.clear();
	delete []m_MaskArray; m_MaskArray = NULL;
}

void CC::InitConfig(int nAreaThreshold)
{
	m_nAreaThreshold = nAreaThreshold;
}

void CC::Process()
{
	Clear();
	if(m_MaskArray == NULL) return;
	
	Binarize(); // if pixel_intensity > 0 --> pixel_intensity = 1;

	KLinkedList eqTable;

	KNode * tmp = NULL;
	KNode * p	= NULL;
	KNode * p2	= NULL;

	int i;
	int	label = 2;
	int	index = 1;
	int	north = 0;
	int	west  = 0;
	int	nWest = 0;
	int	nEast = 0;
	
	int		 * regionLabel = NULL;
	int		 * lookUpTable = NULL;
	long int * regionArea  = NULL;
	
	int	regionNumber;

	int	maxArea;
	int	maxIndex;
	
	int	data=0;

	for(i=0; i<m_height; i++)
	{
		for(int j=0; j<m_width; j++)
		{			
			index = i*m_width+j;
			
			if(m_MaskArray[index] == 1)
			{
				north = 0;
				west  = 0;
				nWest = 0;
				nEast = 0;

				if( i==0 && j != 0  )
				{
					west = m_MaskArray[index-1];
				}
				else if( i!=0 && j == 0 )
				{
					north = m_MaskArray[index-m_width];
					nEast = m_MaskArray[index-m_width+1];
				}
				else if( i!= 0 && j != 0 )
				{
					north = m_MaskArray[index-m_width];
					west  = m_MaskArray[index-1];
					nWest = m_MaskArray[index-m_width-1];

					if( j != m_width-1 )
						nEast= m_MaskArray[index-m_width+1];
				}

/*				if( index/m_width == 0 ) 
				{
					if( index > 0 )
						west = m_MaskArray[index-1];
				}
				else 
				{
					north = m_MaskArray[index-m_width];
					if( index%m_width>0 ) {
						nWest= m_MaskArray[index-m_width-1];
						west = m_MaskArray[index-1];
					}
					if( index%m_width<m_width-1)
						nEast= m_MaskArray[index-m_width+1];
				}
		*/		
				// after finding the neighbour labels
				if ( west > 1 ) {

					m_MaskArray[index] = west;

					if( nWest>1 )
						eqTable.InsertData(west,nWest);

					if( north>1 && nWest<=1 )
						eqTable.InsertData(west,north);

					if( nEast>1 && north<=1 )
						eqTable.InsertData(west,nEast);
				
				}else if( nWest > 1) {
					
					m_MaskArray[index] = nWest;

					if ( north<=1 && nEast>1 )
						eqTable.InsertData(nWest,nEast);
				}else if( north > 1 ) {
					m_MaskArray[index] = north;
				}else if( nEast > 1 ) 
					m_MaskArray[index] = nEast;
				else {
					m_MaskArray[index] = label;
					eqTable.InsertData(label);
					label++;
				}
			}
		}
	}

	regionNumber = eqTable.regionCount;

	if( regionNumber > 0 ) 
	{
		regionLabel  = new int[regionNumber];
		regionArea   = new long int[label];

		for(i=0; i<label; i++ )
			regionArea[i]=0;

		tmp = eqTable.header;
		i=0;
		do
		{
			regionLabel[i]=tmp->data;
			tmp=tmp->ngNext;
			i=i+1;
		}
		while(tmp!=NULL);

		lookUpTable = new int[label];
	
		p  = eqTable.header;
		p2 = p;
		
		do
		{   
			data=p->data;
			do
			{   
				lookUpTable[p2->data] = data;
				p2 = p2->sgNext;
			}
			while(p2 != NULL);
			p  = p->ngNext;
			p2 = p;
		}
		while(p != NULL);		

		for (i=0;i<m_height; i++  ) {
			for (int j=0; j<m_width; j++ )
			{
				index=i*m_width+j;

				if( m_MaskArray[index]>1 ) 
				{
					data=lookUpTable[ m_MaskArray[index] ];
					m_MaskArray[index]=data;

					regionArea[ data ]++;
				}
				else
					m_MaskArray[index]=0;
			}
		}

//		saveBMP("ccl_.bmp",m_width,m_height,m_MaskArray,'g');

		maxArea = regionArea[0];
		maxIndex=0;
		for(i=1; i<label; i++)
		{
			if(regionArea[i]>maxArea)
			{
				maxIndex = i;
				maxArea = regionArea[i];
			}
		}

		int* trueLabelArray = new int[label];
		for(i=0; i<label; i++)
		{
			trueLabelArray[i] = 0;

			KBox newComponent;
			newComponent.ID = i+1;
                        newComponent.bottomRight = CPoint(RAND_MAX, RAND_MAX);
                        newComponent.topLeft     = CPoint(-RAND_MAX, -RAND_MAX);
			m_Components.push_back(newComponent);
		}

		m_ObjectNumber = 0;
		
		int nImSize = m_height*m_width;
		int x, y;

		for(i=0; i<nImSize; i++)
		{
			x = i%m_width;
			y = m_height-i/m_width-1;

			if( regionArea[ m_MaskArray[i] ] < m_nAreaThreshold ) 
				m_MaskArray[i] = 0;
			else
			{ 
				if( trueLabelArray[ m_MaskArray[i] ] == 0 )
				{
					m_ObjectNumber++;

					m_Components[ m_ObjectNumber-1 ].topLeft.x     = x;
					m_Components[ m_ObjectNumber-1 ].topLeft.y     = y;
					m_Components[ m_ObjectNumber-1 ].bottomRight.x = x;
					m_Components[ m_ObjectNumber-1 ].bottomRight.y = y;

					trueLabelArray[ m_MaskArray[i] ] = m_ObjectNumber;
					m_MaskArray[i] = m_ObjectNumber;
				}
				else
				{
					m_MaskArray[i] = trueLabelArray[ m_MaskArray[i] ];

					if( x > m_Components[ m_MaskArray[i]-1 ].bottomRight.x )
						m_Components[ m_MaskArray[i]-1 ].bottomRight.x = x;

					if( x < m_Components[ m_MaskArray[i]-1 ].topLeft.x     )
						m_Components[ m_MaskArray[i]-1 ].topLeft.x     = x;

					if( y > m_Components[ m_MaskArray[i]-1 ].bottomRight.y )
						m_Components[ m_MaskArray[i]-1 ].bottomRight.y = y;
					
					if( y < m_Components[ m_MaskArray[i]-1 ].topLeft.y     )
						m_Components[ m_MaskArray[i]-1 ].topLeft.y     = y;
				}
			}
		}

		while( m_Components.size() != m_ObjectNumber )
			m_Components.pop_back();


        delete [] trueLabelArray;
        trueLabelArray = NULL;
	}
	
	delete []lookUpTable;
	lookUpTable = NULL;

	delete []regionArea;
	regionArea = NULL;

	delete []regionLabel;
	regionLabel = NULL;
}



void CC::OnCalculateHex(int *array, int a)
{
	for(int i=0; i<4; i++)
	{
		array[i] = a%256;	
		a = a/256;
	}
}

//////// private KNode implementations ////////////////////

CC::KNode::KNode()
{
	this->data=0;
	this->sgNext=NULL;
	this->ngNext=NULL;
}

CC::KNode::~KNode()
{

}


KBox::KBox()
{
	ID = 0;
	bottomRight = 0;
	topLeft = 0;
}

KBox::~KBox()
{
}






///////////////////////////////////////////////////////////

///////// private KLinkedList implementations ///////////////////////////////////////////////

CC::KLinkedList::KLinkedList()
{
	this->header = NULL;
	this->regionCount = 0;
}

CC::KLinkedList::~KLinkedList()
{
	KNode* ptr1 = header;
	KNode* ptr2 = header;
	KNode* ptr3 = header;
	
	if( header != NULL ) {
		do 
		{
			do 
			{
				if (ptr2->sgNext != NULL){
					ptr3 = ptr2;
					ptr2 = ptr2->sgNext;
				} else if( ptr1->sgNext != NULL ) {
					delete ptr2;
					if( ptr3 != NULL )
						ptr3->sgNext=NULL;
					ptr2 = ptr1;
					ptr3 = ptr1;
				}
			}
			while(ptr1->sgNext !=NULL);
			
			ptr1=ptr1->ngNext;
			delete ptr2;
			ptr2=ptr1;
			ptr3=ptr1;
		}
		while(ptr1!=NULL);
	}
}

void CC::KLinkedList::InsertData(int data)
{
	KNode *	ptrTemp = new KNode;
	ptrTemp->data=data;
	ptrTemp->ngNext=header;
	header=ptrTemp;
	regionCount++;
}

void CC::KLinkedList::InsertData(int addGroup, int searchGroup)
{
	if ( addGroup != searchGroup ) {
		
		KNode* tmp1 = header;
		KNode* ptrAdd ;
		KNode* ptrSearch ; 

		Search(addGroup,ptrAdd);
		Search(searchGroup,ptrSearch);

		if ( (ptrSearch != NULL) && (ptrAdd != NULL) && (ptrSearch!=ptrAdd) ) {
			
			if ( ptrSearch != header ) {
				
				while( tmp1->ngNext != ptrSearch )
					tmp1=tmp1->ngNext;
				
				tmp1->ngNext=ptrSearch->ngNext;
			}
			else{
				header=ptrSearch->ngNext;
			}
			
			while( ptrAdd->sgNext != NULL )
				ptrAdd=ptrAdd->sgNext;
			
			ptrAdd->sgNext=ptrSearch;	
		}
	}

}



void CC:: KLinkedList::Search(int data, CC::KNode* &p )
{
	KNode* ptr1 = header;
	KNode* ptr2 = header;
	
	do 
	{
		do 
		{
			if (ptr2->data==data){
				p=ptr1;
				return;
			}
			ptr2=ptr2->sgNext;
		}
		while(ptr2!=NULL);
		
		ptr1=ptr1->ngNext;
		ptr2=ptr1;
	}
	while(ptr1!=NULL);
	
	p=ptr1;
}


/////////////////////////////////////////////////////////////////////////////////////////////


void CC::SetMask(double *mask, int width, int height)
{
	if( m_MaskArray != NULL )
		delete []m_MaskArray;

	m_width     = width;
	m_height	= height;
	
	int size = width*height;
	
    m_MaskArray = new int[size];

	for(int i=0; i<size; i++)
	{
		m_MaskArray[i] = (int)mask[i];
	}
}

void CC::Binarize()
{
	if( m_MaskArray == NULL )
		return;

	int index, size = m_width*m_height;
	
	for(index=0; index<size; index++ )
	{	
		if( m_MaskArray[index] > 0 )
			m_MaskArray[index] = 1;
	}
}

int* CC::GetOutput()
{
	return m_MaskArray;
}

void CC::Clear()
{
	m_Components.clear();
	m_ObjectNumber = 0;
}

void CC::Merge()
{

	int sz = m_Components.size();

	if( sz == 1 || sz == 0 ) return;

//	if(sz == 2)
//		int tmp = 0;
	
	CPoint* center = new CPoint[sz];
	CPoint* dim    = new CPoint[sz];

	for( int i=0; i<sz; i++ )
	{
		center[i].x = (m_Components.at(i).bottomRight.x + m_Components.at(i).topLeft.x) / 2; 
		center[i].y = (m_Components.at(i).bottomRight.y + m_Components.at(i).topLeft.y) / 2;

		dim[i].x    = (center[i].x - m_Components.at(i).topLeft.x);
		dim[i].y    = (center[i].y - m_Components.at(i).topLeft.y);

		KBox test = m_Components.at(i);
	}

        int *equalTable = new int[sz*sz];
	
        for(int i=0; i<sz*sz; i++)
	{ 
		equalTable[i] = 0;
	}

	int *labels = new int[sz];
	int mergeDist = 5;
	int * mergeCheck = new int[sz];
        for( int i=0; i<sz; i++ )
	{
		labels[i] = 0;
		mergeCheck[i] = 1;
		equalTable[i*sz+i] = 1;

		for( int j=i+1; j<sz; j++ )
		{

			if( (abs(center[i].x - center[j].x) - dim[i].x - dim[j].x) < mergeDist && 
				(abs(center[i].y - center[j].y) - dim[i].y - dim[j].y) < mergeDist )
				equalTable[i*sz+j]=1;

		}
		//equallari yaz;

	}

        for( int i=sz-2; i>=0; i-- )
	{
		for( int j=i+1; j<sz; j++ )
		{
			// check if rows i and j are equal
			if( equalTable[i*sz+j] > 0 )
			{
				// merge two rows
				for( int k=j+1; k<sz; k++)
				{
					equalTable[i*sz+k] += equalTable[j*sz+k];
				}
			}
		}
	}

	int mx = 0;
        for( int i=0; i<sz; i++)
	{
		for( int j=0; j<=i; j++  )
		{
			if( equalTable[j*sz+i] >0 )
			{
				if( j > mx )
				{
					mx++; // finds the object number;
					labels[i] = mx;
				}
				else
					labels[i] = j;
				break;
			}
		}
	}

	if( m_ObjectNumber == (mx+1) ) return;


	int minx,maxx,miny,maxy;
	std::vector<KBox> output;

	// recalculate merged box boundaries
        for( int i=0; i<sz; i++ )
	{
		if(mergeCheck[i] == 0)
			continue;

		minx = m_Components.at(i).topLeft.x    ;		
		maxx = m_Components.at(i).bottomRight.x;
		miny = m_Components.at(i).topLeft.y    ;
		maxy = m_Components.at(i).bottomRight.y;

		for( int j=i+1; j<sz; j++  )
		{
			if( equalTable[i*sz+j] < 1 ) 
				continue;

			mergeCheck[j] = 0;

			if( minx > m_Components.at(j).topLeft.x     ) 
				minx = m_Components.at(j).topLeft.x    ;
			if( maxx < m_Components.at(j).bottomRight.x ) 
				maxx = m_Components.at(j).bottomRight.x;
			if( miny > m_Components.at(j).topLeft.y     ) 
				miny = m_Components.at(j).topLeft.y    ;
			if( maxy < m_Components.at(j).bottomRight.y ) 
				maxy = m_Components.at(j).bottomRight.y;
		}
		
		KBox temp;
		temp.ID = i;
		temp.bottomRight = CPoint(maxx,maxy);
		temp.topLeft     = CPoint(minx,miny);
		output.push_back(temp);
	}

	m_ObjectNumber = mx+1; // update object number

	// re-label the image intensities.

        for( int i=0; i<m_width*m_height; i++ )
	{
		if( m_MaskArray[i] == 0 ) continue;
		m_MaskArray[i] = labels[ m_MaskArray[i]-1 ]+1;
	}

	m_Components.clear(); m_Components = output;
	delete []labels;
	delete []mergeCheck;
	delete []equalTable;
}
}
