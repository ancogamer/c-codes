#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

int main() {
	int N, K, I;
	char C;
	
	scanf("%d ", &N);
	int result [N];
	for (int i = 0; i < N; i++) {
        
		scanf("%d ", &K);
		int pila[K]; 
        int filha[K];
        int countP=0;
        int countF=0;
        int qtdInsert=0;
        int removedCount=0;
        int removedCountP=0;
        int removedCountF=0;
        int keepP=1, keepF=1;
		for (int j = 0; j < K; j++) {

			scanf("%c %d ", &C, &I);
            if (C =='i'){
               
                if (keepP){
                    pila[qtdInsert-removedCountP]=I;
                } 
                if (keepF){
                    filha[qtdInsert]=I;
                }
                if ((keepP) || (keepF)){
                    qtdInsert++;      
                }
                continue;                                                                      
            }
            
            if (C =='r'){
                // caso tentar remover mais do que inseriu
                if (qtdInsert == removedCount){
                    continue;
                }
         
               
                removedCountP++;
                if ((keepP)&&(pila[qtdInsert-removedCountP]==I)){
                    countP++;
                }else{ // invalidando a pila;
                    keepP=0;
                    countP=0;
                }
    
                if ((keepF)&&(filha[removedCountF]==I)){
                    removedCountF++;
                    countF++;
                }else{ // invalidando a filha;
                    keepF=0;
                    countF=0; 
                }
                
                
                if  ((pila[j-1-removedCountP]) || (filha[removedCountF]==I)){
                    // alguma coisa foi removida
                    removedCount++;
                }
        
            }
            
		}

		if (countP > countF){
            result[i] = 0;
            continue;
        }
        if ((countP ==0) && (countF ==0)){
            result[i] = 3;
            continue;
        }
        if (countP == countF){
            result[i] = 1;
            continue;
        }
        if (countP < countF){
            result[i] = 2;
            continue;
        }

	}
    
    for (int i = 0; i < N; i++) {
        switch (result[i])
        {
            case 0:
                printf("pilha\n");
                break;
            case 1: 
                printf("indefinido\n");
                break;
            case 2:
                printf("fila\n");
                break;
            default:
                printf("impossivel\n");
                break;
        }
    }

}