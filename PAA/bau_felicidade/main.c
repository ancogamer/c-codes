#include <stdio.h>
#include <stdlib.h>

int main(){	
	int N; // qtd vagas  
	int somaPeso;


	// sempre dividir por 2 pessoas.
    scanf("%d ",&N); 
	int pesos[N];
	for (int i=0; i<N;i++){
		scanf("%d ",&pesos[i]);
		somaPeso+=pesos[i];
	}
	int mod=somaPeso%2;
	printf("%d",mod);

	return 0;

} 
