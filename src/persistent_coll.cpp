#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include "cuda_runtime.h"
#include <unistd.h>

#include "persistent_coll.h"

double t[2], duration[2];

int OSC_A2A(const void* send_buffer, int size, int rank, size_t seg_size, MPI_Win *win, myinfo *P_info)
{
    int step, sendto;
    for ( step = 1; step < size + 1; step++ ){
        sendto = (( rank + step) % size );
        void * tmpsend = (char*)send_buffer + sendto * seg_size;

        //printf("\n[proc %d]R[%d] OSC A2A: sends to %d, scount %d, from tmpsend %p\n", getpid(), rank, sendto, seg_size, tmpsend);
        MPI_Put( tmpsend,
                 seg_size,
                 MPI_BYTE,
                 sendto,
                 rank * seg_size,
                 seg_size,
                 MPI_BYTE,
                 *win );
    }
    MPI_Win_fence(0, *win);
    return 0;
}

int OSC_A2AV(const void* send_buffer, void* recv_buffer, const int* send_counts, 
             const int* send_disp,
             myinfo *P_info)
{
    int step, sendto, recv_disp, seg_size;
    for ( step = 1; step < P_info->nprocs + 1; step++ ){

        sendto = (( P_info->me + step) % P_info->nprocs );

        seg_size = send_counts[sendto] * P_info->datatype_size;

        void * tmpsend = (char*)send_buffer + (send_disp[sendto] * P_info->datatype_size); 

        recv_disp = (P_info->recv_disp_arry[(sendto * P_info->nprocs ) + P_info->me]) * P_info->datatype_size;

        //printf("\n[proc %d]R[%d] OSC A2AV: sends to %d, scount %d, send_disp %d, from tmpsend %p, rdisp %d, win %p\n", getpid(), P_info->me, sendto, send_counts[sendto], send_disp[sendto], tmpsend, recv_disp, P_info->win);

        if( sendto != P_info->me ){
            MPI_Put( tmpsend,                           // origin_addr
                     seg_size,                          // origin_count
                     MPI_BYTE,                          // origin_datatype
                     sendto,                            // target_rank
                     recv_disp,                         // target_disp
                     seg_size,                          // target_count
                     MPI_BYTE,                          // target_datatype
                     *(P_info->win) );                  // MPI_Win    
        }else{
            cudaMemcpy( (void*)(recv_buffer + recv_disp), (void*)tmpsend, seg_size, cudaMemcpyDeviceToDevice);
        }
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
int MPI_Persistent_Init(const int* send_counts_addr, void* recv_buffer, const int* recv_disp_addr, MPI_Comm comm, myinfo *P_info)
{
    //printf("\n[proc %d] Before P_init: has recv_buffer %p, recv_total %d, P_info->nprocs %d\n", getpid(), recv_buffer,P_info->recv_total, P_info->nprocs);

    if( 0 == P_info->P_inited ){
        // Exchanging information with peers within the same P_info->me subcomm
        MPI_Allgather( send_counts_addr, 1, MPI_INT,
                       P_info->recv_elements, 1, MPI_INT,
                       comm );
        MPI_Allgather( recv_disp_addr, P_info->nprocs, MPI_INT,
                       P_info->recv_disp_arry, P_info->nprocs, MPI_INT,
                       comm );
        MPI_Barrier(comm);
/*
        int i,j;
        for(i = 0; i < P_info->nprocs; i++){
            int sendto = ((P_info->me + i) % P_info->nprocs );
            printf("R[%d] has recv_disp[%d][%d] = %d\n", P_info->me, i, j, P_info->recv_disp_arry[(i * P_info->nprocs ) + j]);
            printf("R[%d] has recv_disp from %d to me: %d\n", P_info->me, sendto, P_info->recv_disp_arry[(sendto *  P_info->nprocs) +  P_info->me]);
            //}
        }
        printf("\n$$$ Persistent_Init before inint: R[%d] from comm %p, has recv_buffer %p, size %d, P_info %p, prev_buffer %p, P_info->win %p, P_info->P_inited %d\n", P_info->me, comm, recv_buffer, size, P_info, P_info->prev_recv, P_info->win, P_info->P_inited);
*/

        P_info->P_inited = 1;
        MPI_Win_create(recv_buffer, P_info->recv_total, 1, MPI_INFO_NULL, comm, (P_info->win));
    }
    MPI_Win_fence( 0, *(P_info->win) );

    return 0;
}

int MPI_Persistent_start(const void* send_buffer, void* recv_buffer, const int* send_counts, const int* send_disp, myinfo *P_info)
{
    int size = P_info->nprocs;
    int rank = P_info->me;
    int seg_size = send_counts[0] * P_info->datatype_size;
    
    // All-to-all operation
    //return OSC_A2A(send_buffer, size, rank, seg_size, P_info->win, P_info);

    // All-to-allv operation
    return OSC_A2AV(send_buffer, recv_buffer, send_counts, send_disp, P_info);
}

void MPI_Persistent_stop(myinfo *P_info, void* recv_buffer)
{
    MPI_Win_detach( *(P_info->win), recv_buffer );
    MPI_Win_fence( 0, *(P_info->win) );
}

void MPI_Persistent_end(myinfo *P_info)
{
    if( NULL != P_info ){
        if( NULL != P_info->recv_elements ) { free(P_info->recv_elements); P_info->recv_elements = NULL; }
        if( NULL != P_info->recv_disp_arry ) { free(P_info->recv_disp_arry); P_info->recv_disp_arry = NULL; }
        if( NULL != P_info->win) { MPI_Win_free( (P_info->win) ); }
        if( NULL != P_info->prev_recv ) { P_info->prev_recv = NULL; }
        if( NULL != P_info ) { free(P_info); P_info = NULL; }
    }
}

int MPI_OSC_Alltoallv(const void *sendbuf, const int sendcounts[],
                      const int sdispls[], MPI_Datatype sendtype,
                      void *recvbuf, const int recvcounts[],
                      const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                      myinfo *P_info)
{
    MPI_Persistent_Init(sendcounts, recvbuf, rdispls, comm, P_info);

    MPI_Persistent_start(sendbuf, recvbuf, sendcounts, sdispls, P_info);

    // 4 P_info structs (per proc) will be freed in destructor
    return 0;
}

void Timer_start()
{
    t[0] = MPI_Wtime();
}

void Ttimer_add()
{
    t[0] += MPI_Wtime();    
}

void Timer_end()
{
    t[1] = (MPI_Wtime() - t[0]);
}

