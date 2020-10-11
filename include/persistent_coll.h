#ifndef PERSISTENT_COLL_H
#define PERSISTENT_COLL_H

#include "mpi.h"

typedef struct myinfo{
    int*        recv_elements;
    int*        recv_disp_arry;
    MPI_Win*    win;
    bool        P_inited;
    void*       prev_recv;
}myinfo;

// For Persistent A2AV
extern int MPI_OSC_Alltoallv(const void* send_buffer, int nprocs, int me, const int* send_counts, 
                             const int* send_disp, const int* recv_disp, int datatype_siz,
                             myinfo *P_info);

extern int MPI_Persistent_Init(int me, int nprocs, const int* send_size, int recv_total, void* recv_buffer, 
                               const int* recv_disp, MPI_Comm comm, int datatype_size, myinfo *P_info);

extern int MPI_Persistent_start(const void* send_buffer, int nprocs, int me, const int* send_counts, 
                                const int* send_disp, int datatype_size, myinfo *P_info);

extern void MPI_Persistent_stop(myinfo *P_info, void* recv_buffer);

extern void MPI_Persistent_end(myinfo *P_info);

extern int get_segdsize (struct ompi_datatype_t *sdtype, int scount);
#endif
