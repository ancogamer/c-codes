#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
M E N C 
M = numero de salas
E = ligações entre elas corredores
N = ligações entre elas tubulações
C = consultas

U E V = inteiros 
D = float


Numero de Salas, Numero de corredores, quantidade de tubulações, numero de consultas 
U V D | U V D | U V D -> usa 3 variaveis para descrever a distancia entre as salas, sala 1, sala 2, distancia.
indica vertices de peso 1 entre as salas 
consulta, sala aonde o impostor foi visto 
consulta, pode vir varias*/
int main(){	
	int M;   // number of rooms
	int E; 	 // number of hallways
    int N; 	 // number of vents
	int C;   // where the sus was saw

	scanf("%d %d %d %d",&M,&E,&N,&C);
	int U=0;
	int V=0;
	float D=0;
	float adjacentMatrix[M][M];
	float adjacentMatrixSUS[M][M];
	adjacentMatrix[0][0]=0;
	adjacentMatrixSUS[0][0]=0;

	for (int c = 0;c<E *3;c+=3){
		scanf("%d %d %f ",&U,&V,&D); 
		adjacentMatrix[U][V]=D;
		adjacentMatrixSUS[U][V]=D;
	}
	// zerando valores das ventilações para o sus
	for (int i = 0;i<N *2;i+=2){
		scanf("%d %d",&U,&V);
		adjacentMatrixSUS[U][V]=0;
	}

	int room=0;
	// defeat  victory
	
	float soma=0;
	float somaSus=0;
	int result[C];
	for (int i =0; i<C;i++){
		scanf("%d",&room);
		soma=adjacentMatrix[0][room];
		if(soma == 0 ){ // soma = 0 sinal que não existe conexão entre o vertice 0 e aonde o Sus foi visto
			for (int r=room;r>0;r--){ 
					if (adjacentMatrix[0][r]!=0){
						if (adjacentMatrix[r][room]!=0){
							soma=adjacentMatrix[r][room];
						}
					}
		}
		printf("victory\n");
		 
	}



	// shows the adjacent matrix
		printf("\n");
	for (int i = 0;i<M+1;i++){
		for (int j = 0;j<M+1;j++)
			printf("%.2f ",adjacentMatrix[i][j]);
		printf("\n");
	}
	
	return 0;

} 
