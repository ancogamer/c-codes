#include <stdio.h>
#include <stdlib.h>

int main(){	
	int N=0;  
	int somaPeso=0;
	int peso;
	
    scanf("%d ",&N); 
	for (int i=0; i<N;i++){
		scanf("%d ",&peso);
		somaPeso+=peso;
	}
	printf("%d",somaPeso%2);
	return 0;

} 
