#include <stdio.h>
#include <string.h>
#include <cstdlib>

#include "persistent_coll.h"

int MPI_OSC_Alltoall(const void* send_buffer, int size, int rank, size_t seg_size, MPI_Win *win)
{
    int step, sendto;
    for ( step = 1; step < size + 1; step++ ){
        sendto = (( rank + step) % size );
        void * tmpsend = (char*)send_buffer + sendto * seg_size;

        MPI_Put( tmpsend,
                 seg_size,
                 MPI_BYTE, sendto,
                 rank * seg_size,
                 seg_size,
                 MPI_BYTE,
                 *win );
    }
    MPI_Win_fence(0, *win);
    return 0;
}

int MPI_OSC_Alltoallv(const void* send_buffer, int nprocs, int me, const int* send_counts, 
                      const int* send_disp, int datatype_size,
                      myinfo *P_info)
{
    int step, sendto, recv_disp, seg_size;
    for ( step = 1; step < nprocs + 1; step++ ){
        sendto = (( me + step) % nprocs );
        void * tmpsend = (char*)send_buffer + send_disp[sendto] * datatype_size;

        seg_size = send_counts[sendto] * datatype_size;

        recv_disp = (P_info->recv_disp_arry[(sendto * nprocs ) + me]) * datatype_size;

        //printf("\n$$$ OSC A2AV: R[%d] has %d peers, sends to %d scount %d, sdips %d, rdisp %d, win %p\n", me, nprocs, sendto, send_counts[sendto], send_disp[sendto], recv_disp, P_info->win);

        MPI_Put( tmpsend,                          // origin_addr
                 seg_size,                          // origin_count
                 MPI_BYTE,                          // origin_datatype
                 sendto,                            // target_rank
                 recv_disp,                         // target_disp
                 seg_size,                          // target_count
                 MPI_BYTE,                          // target_datatype
                 *(P_info->win) );                   // MPI_Win    
    }
    MPI_Win_fence(0, *(P_info->win));

    return 0;
}

int get_segdsize (struct ompi_datatype_t *sdtype, int scount)
{
    char datatype[20];
    int  retlen  =  0;

    MPI_Type_set_name( MPI_DOUBLE, "DOUBLE" );
    MPI_Type_set_name( MPI_C_DOUBLE_COMPLEX, "DOUBLE_COMPLEX" );
    MPI_Type_set_name( MPI_FLOAT, "FLOAT" );
    MPI_Type_set_name( MPI_INT, "INT" );
    MPI_Type_get_name( sdtype, datatype, &retlen );

    if ( strcmp( datatype, "DOUBLE") == 0 ) return scount * sizeof(double);
    if ( strcmp( datatype, "DOUBLE_COMPLEX") == 0 ) return scount * sizeof(double _Complex);
    if ( strcmp( datatype, "FLOAT") == 0 ) return scount * sizeof(float);
    if ( strcmp( datatype, "INT") == 0 ) return scount * sizeof(int);

    return -1;
}

// Init for Persistent A2AV; exchange information and create window for one-sided communication A2AV
int MPI_Persistent_Init(int me, int nprocs, const int* send_size, int recv_total, void* recv_buffer, const int* recv_disp, MPI_Comm comm, int datatype_size, myinfo *P_info)
{


    if( 0 == P_info->P_inited ){
        // Exchanging information with peers in the same comm
        MPI_Barrier(comm);
        MPI_Allgather( send_size, 1, MPI_INT,
                       P_info->recv_elements, 1, MPI_INT,
                       comm );
        MPI_Barrier(comm);

        MPI_Allgather( recv_disp, nprocs, MPI_INT,
                       P_info->recv_disp_arry, nprocs, MPI_INT,
                       comm );
        MPI_Barrier(comm);

/*
        int i,j;
        for(i = 0; i < nprocs; i++){
            for(j = 0; j < nprocs; j++){
                printf("R[%d] has recv_disp[%d][%d] = %d\n", me, i, j, P_info->recv_disp_arry[(i* nprocs ) + j]);
                printf("R[%d] has recv_disp from %d to me: %d\n", me, i, P_info->recv_disp_arry[(i* nprocs ) + me]);
            }
        }
*/
        int size = recv_total * datatype_size;
        //printf("\n$$$ Persistent_Init: R[%d] from comm %p, has recv_buffer %p, size %d, P_info %p, prev_buffer %p, P_info->win %p, P_info->P_inited %d\n", me, comm, recv_buffer, size, P_info, P_info->prev_recv, P_info->win, P_info->P_inited);

        MPI_Win_create(recv_buffer, recv_total * datatype_size, 1, MPI_INFO_NULL, comm, (P_info->win));
        //MPI_Win_create_dynamic( MPI_INFO_NULL, comm, (P_info->win) );
        //MPI_Win_attach( *(P_info->win), recv_buffer, size);
        MPI_Win_fence( 0, *(P_info->win) );

        P_info->P_inited = 1;        
        P_info->prev_recv = recv_buffer;

    }else if( P_info->prev_recv != recv_buffer ){

        MPI_Persistent_stop( P_info, P_info->prev_recv );

        int size = recv_total * datatype_size;
        //printf("$$$ Persistent Init: R[%d] had old recv addr %p and new recv addr %p, size %d, reattached to win %p\n", me, P_info->prev_recv, recv_buffer, size, P_info->win);

        //MPI_Win_attach( *(P_info->win), recv_buffer, size);
        MPI_Win_create(recv_buffer, recv_total * datatype_size, 1, MPI_INFO_NULL, comm, (P_info->win));
        MPI_Win_fence( 0, *(P_info->win) );

        P_info->P_inited = 1;        
        P_info->prev_recv = recv_buffer;

    }else{
        printf("$$$ Persistent Init: R[%d] already inited P_info %p and win %p\n", me, P_info, P_info->win);
    }
    return 0;
}

int MPI_Persistent_start(const void* send_buffer, int nprocs, int me, const int* send_counts, const int* send_disp, int datatype_size,  myinfo *P_info)
{
    return MPI_OSC_Alltoallv(send_buffer, nprocs, me, send_counts, send_disp, datatype_size, P_info);
}

void MPI_Persistent_stop(myinfo *P_info, void* recv_buffer)
{
    MPI_Win_detach( *(P_info->win), recv_buffer );
    MPI_Win_fence( 0, *(P_info->win) );
}

void MPI_Persistent_end(myinfo *P_info)
{
    if( NULL != P_info->recv_elements ) { free(P_info->recv_elements); P_info->recv_elements = NULL; }
    if( NULL != P_info->recv_disp_arry ) { free(P_info->recv_disp_arry); P_info->recv_disp_arry = NULL; }
    MPI_Win_free( (P_info->win) );
    if( NULL != P_info ) { free(P_info); P_info = NULL; }
}


