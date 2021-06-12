//Ramon quer saber a menor nota tirada //dentre um determinado numero de pessoas.

#include <stdio.h>
#include <stdlib.h>
typedef struct Notas {
    float  value; // conteúdo    
    struct Notas *dir;
    struct Notas *esq;
      
} Notas;

float *NotasArray;

void insereArvore(Notas** t, float num){
    /* Essa função insere os elementos de forma recursiva */
    if(*t == NULL)
    {
        *t = (struct Notas *) malloc(sizeof(Notas)); /* Aloca memória para a estrutura */
        (*t)->esq = NULL; /* Subárvore à esquerda é NULL */
        (*t)->dir = NULL; /* Subárvore à direita é NULL */
        (*t)->value = num; /* Armazena a informação */
    } else {
        if(num < (*t)->value) /* Se o número for menor então vai pra esquerda */
        {
            /* Percorre pela subárvore à esquerda */  
            insereArvore(&(*t)->esq, num);
        }
        if(num > (*t)->value) /* Se o número for maior então vai pra direita */
        {
            /* Percorre pela subárvore à direita */
            insereArvore(&(*t)->dir, num);
        }
    }
}
Notas* criaNota()
{
  /* Uma árvore é representada pelo endereço do nó raiz,
     essa função cria uma árvore com nenhum elemento,
     ou seja, cria uma árvore vazia, por isso retorna NULL. */
  return NULL;
}

void ordem(struct Notas* x,int* count){
    if (x!=NULL){
        ordem(x->dir,count);
        NotasArray[*count]=x->value;
        *count=*count+1;
	    free(x);
        //printf("%f\n",x->value);
	    ordem(x->esq,count++);
    }
}

int main(int argc, char *argv[]){
	int N; // qtd vagas  
	int K; // qtd pessoas per vaga
    int C; // numeros de candidatos que concorrem à vaga. 
 
    
    scanf("%d",&N);
    float result[N];
    
    for (int i=0; i<N;i++){
        scanf("%d %d",&K,&C);
        Notas* arv = criaNota(); 
        NotasArray= (float*) malloc(C * sizeof(float));
        float notas; // notas dos candidatos; 
        for (int j=0;j<C;j++){
            scanf("%f",&notas);
            insereArvore(&arv,notas);
        } 
        int count = 0;
        ordem(arv,&count);    
        // em questão, posição no array é = K-1, pq começa em array começa em 0.
        //printf("%.2f\n",NotasArray[(K-1)*C]);
        result[i]=NotasArray[(K-1)*C];
        free(NotasArray);
        free(arv);
    }
    for (int i=0; i<N;i++){
        printf("%.2f\n",result[i]);
    } 
    

	return 0;

} 
