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

int main(int argc, char** argv){	
    int N; // qtd vagas  
	int K; // qtd pessoas per vaga
    int C; // numeros de candidatos que concorrem à vaga. 
 
    

    FILE *fp;
    char **buf;
    char b;
 
    fp = fopen(argv[1], "r");
    int i =0;
    
    float **result;
    
    Notas* arv = criaNota(); 
    
    while((b = fgetc(fp)) != EOF){
        buf=&b;
        if ((buf[i]!="\n")&&(i-1!=0)){
            printf("to aqui");
            if (i==1){
                K=buf[i];
            }
            //scanf("%d %d",&K,&C);
            NotasArray= (float*) malloc(C * sizeof(float));
            printf("to aqui2"); 
            
            float notas=b; // notas dos candidatos; 
            //for (int j=0;j<C;j++){
                scanf("%f",&notas);
                insereArvore(&arv,notas);
            //} 
            printf("to aqui3"); 
               
            // em questão, posição no array é = K-1, pq começa em array começa em 0.
            //printf("%.2f\n",NotasArray[(K-1)*C]);
        }else{
            int count = 0;
            ordem(arv,&count); 
            printf("to aqui1");
            result[i]=&NotasArray[(i-2)*K];
            free(NotasArray);
            free(arv);
            i=0;
        }
    }
    
    
    N=buf[0];

    for (int i=0; i<N-1;i++){
        printf("%.2f\n",result[i]);
    } 
    

	return 0;

} 
