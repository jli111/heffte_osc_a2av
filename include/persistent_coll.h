#ifndef PERSISTENT_COLL_H
#define PERSISTENT_COLL_H

#include "mpi.h"

typedef struct myinfo{
    int         nprocs;
    int         me;
    int         datatype_size;

    int*        recv_elements;
    int*        recv_disp_arry;

    MPI_Win*    win;
    bool        P_inited;

    void*       prev_recv;
    int         recv_total;
    int         prev_recv_total;
    double      time[10];

    MPI_Datatype sendtype;
    MPI_Datatype recvtype;

}myinfo;

// For Persistent A2AV
extern int OSC_A2AV(const void* send_buffer, void* recv_buffer, const int* send_counts, 
                    const int* send_disp, const int* recv_disp,
                    myinfo *P_info);

extern int MPI_Persistent_Init(const int* send_counts_addr, void* recv_buffer, 
                               const int* recv_disp_addr, MPI_Comm comm, myinfo *P_info);

extern int MPI_Persistent_start(const void* send_buffer, void* recv_buffer, const int* send_counts, 
                                const int* send_disp, myinfo *P_info);

extern void MPI_Persistent_stop(myinfo *P_info, void* recv_buffer);

extern void MPI_Persistent_end(myinfo *P_info);

extern int MPI_OSC_Alltoallv(const void *sendbuf, const int sendcounts[],
                         const int sdispls[], MPI_Datatype sendtype,
                         void *recvbuf, const int recvcounts[],
                         const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                         myinfo *P_info);

extern int get_segdsize (struct ompi_datatype_t *sdtype, int scount);

extern void   Timer_start();
extern void   Timer_add();
extern void   Timer_end();

#endif
