#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h> 


typedef struct C {
    char n[10];
    int nu;  
    int peso;
    struct C* dir;
    struct C* esq;
      
} C;

int max(int a, int b){
    return (a > b)? a : b;
}

int getPeso(struct C *c){
    if (c == NULL)
        return 0;
    return c->peso;
}


int diferencapeso(struct C *c){
    if (c == NULL)
        return 0;

    int esqR =0;
    int dirR =0;    
    if (c->esq != NULL)
        esqR=c->esq->peso;
    if (c->dir != NULL)
        dirR=c->dir->peso;    

    return esqR-dirR;
}


C * minValorNo(struct C* c){
    struct C* atual = c;
    for(;;){
        if (atual->esq == NULL){
            break;
        }
        atual = atual->esq;
    }
    return atual;
    
}

C* giraDireita(struct C *c){
    struct C *x = c->esq;
    struct C *Coringa = x->dir;
    x->dir = c;
    c->esq = Coringa;
    c->peso = max(getPeso(c->esq), getPeso(c->dir))+1;
    x->peso = max(getPeso(x->esq), getPeso(x->dir))+1;
 
    return x;
}

C* giraEsquerda(struct C *c){
    struct C *y = c->dir;
    struct C *Coringa = y->esq;
    y->esq = c;
    c->dir = Coringa;
    c->peso = max(getPeso(c->esq), getPeso(c->dir))+1;
    y->peso = max(getPeso(y->esq), getPeso(y->dir))+1;
 
    return y;
}

int found =0;
C* rC(struct C* c,char n[10]){
    if (c == NULL){
        return c;
    }
    if (strcmp(n, c->n) < 0 ){
        c->esq = rC(c->esq, n);
    }else if (strcmp(n, c->n) > 0 ){
        c->dir = rC(c->dir, n);
    }else if (strcmp(n, c->n) == 0 ){
        found =1;
        if( (c->esq == NULL) || (c->dir == NULL) )
        {
            struct C *aux = c->esq ? c->esq : c->dir;
            if (aux == NULL)
            {
                aux = c;
                c = NULL;
            }else{  
                *c = *aux; 
            }
            free(aux);
        }
        else
        {
            C* aux = minValorNo(c->dir);
            
            strcpy(c->n,aux->n);
            c->nu = aux->nu;
            c->dir = rC(c->dir, aux->n);
        }
    }
    if (c == NULL){
       return c;
    }
    
    
    c->peso = 1 + max(getPeso(c->esq),getPeso(c->dir));
 
    int balance = 0;
    if (c != NULL){
       balance= getPeso(c->esq) - getPeso(c->dir);
    }
 
    if (balance > 1 && diferencapeso(c->esq) >= 0){
        return giraDireita(c);
    }

    if (balance > 1 && diferencapeso(c->esq) < 0){
        c->esq =  giraEsquerda(c->esq);
        return giraDireita(c);
    }

    if (balance < -1 && diferencapeso(c->dir) <= 0){
        return giraEsquerda(c);
    }
    if (balance < -1 && diferencapeso(c->dir) > 0){
        c->dir = giraDireita(c->dir);
        return giraEsquerda(c);
    }
 
    return c;
}

C* iC(C* t, int nu,char n[10]){
    if(t == NULL)
    {
        C* c = (struct C *) malloc(sizeof(C));
        c->esq = NULL; 
        c->dir = NULL; 
        c-> nu= nu;
        strcpy(c->n,n); 
        c-> peso= 1;
        return c; 
    }
    

    if(strcmp(n, t->n) < 0 ){
        t->esq=iC(t->esq, nu,n);
    }else if(strcmp(n, t->n) == 0){
        printf("Contatinho ja inserido\n");
        return t;
    }else{
        t->dir=iC(t->dir, nu,n);
    }

    t->peso = 1 + max(getPeso(t->esq),getPeso(t->dir));

    int balance = 0;
    if (t != NULL){
       balance= getPeso(t->esq) - getPeso(t->dir);
    }
   
    if (balance > 1 && strcmp(n,t->esq->n) < 0 ){
        return giraDireita(t);
    }
   
    if (balance < -1 && strcmp(n,t->dir->n) >0 ){
        return giraEsquerda(t);
    }

    if (balance > 1 && strcmp(n,t->esq->n) >0 ) 
    {
        t->esq =  giraEsquerda(t->esq);
        return giraDireita(t);
    }
 
  
    if (balance < -1 &&  strcmp(n,t->dir->n) >0 ){
        t->dir = giraDireita(t->dir);
        return giraEsquerda(t);
    }
    return t;
}


C* cF(C* c, char n[10]){
    if (c == NULL){
        return c;
    }
    struct C* atual = c;
    for (;atual!=NULL;){
        if(strcmp(n, atual->n) == 0){
            break;
        }else if(strcmp(n, atual->n) < 0){
            atual = atual -> esq;
        }else{
            atual = atual -> dir;
        }
    }
    return atual; 
}

int main(){
    struct C *c = NULL;
    char co;
   
    C* cf=NULL;
    for (;;){
        int nu;
        char na[10];
        scanf("%c ",&co);
        switch (co)
        {
        case 'I':
            scanf(" %s %d",na,&nu);
            c = iC(c,nu,na);
            break;
        case 'P':
            scanf(" %s",na);
            cf=cF(c,na);
            if (cf==NULL){
                printf("Contatinho nao encontrado\n");
                continue;
            }
            printf("Contatinho encontrado: telefone %d\n",cf->nu);
            break;
       
        case 'R':
            scanf(" %s",na);
            c=rC(c,na);
            if (found == 0 ){
                printf("Operacao invalida: contatinho nao encontrado\n");
                continue;
            }
            found =0;
            break;
        case 'A':
            scanf(" %s %d",na,&nu);
            cf=cF(c,na);
            if (cf==NULL){
                printf("Operacao invalida: contatinho nao encontrado\n");
                continue;
            }
            cf->nu=nu;
            break;

        case '0':
            free(c);
            return 0;
        }
    }
   

	return 0;
} 

