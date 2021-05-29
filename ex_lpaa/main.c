//Ramon quer saber a menor nota tirada //dentre um determinado numero de pessoas.

#include <stdio.h>

int main(){
	int N; // qtd vagas  
	int K; // qtd pessoas per vaga
    int C; // numeros de candidatos que concorrem à vaga. 
    


    scanf("%d",&N);
    float maior;
    for (int i=0; i<N;i++){
        scanf("%d %d",&K,&C);
        float notas[C]; // notas dos candidatos; 
        for (int j=0;j<C;j++){
            scanf("%f",&notas[i]);   
            if (j!=0) {
                if (notas[j] < notas[j-1]) {
                    int aux = notas[j-1];
                    aux = notas[j-1];
                    notas[j-1]=notas[j];
                    notas[j]=aux;
                    maior = notas[j];
                }
            }
        }
        for (int j=0;j<K;j++){
            printf("%.2f",notas[j]);   
        }
    }

	return 0;

} 
