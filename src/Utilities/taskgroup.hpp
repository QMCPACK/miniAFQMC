//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef AFQMC_TASK_GROUP_H
#define AFQMC_TASK_GROUP_H

#include<vector>
#include<string>
#include<map>
#include<ctime>
#include<sys/time.h>
#include<cstdlib>
#include<ctype.h>
#include<algorithm>
#include<iostream>
#include<ostream>
#include <mpi.h>

#include"Configuration.h"

namespace qmcplusplus
{

// sets up communicators and task groups
// Various divisions are setup:
//   1. head_of_nodes: used for all shared memory setups
//   2. breaks global comm into groups of ncores_per_TG x nnodes_per_TG
//      and sets up appropriate communicators     
//   Right now does not allow communications outside the TG.
//   This option must be enabled in order to implement algorithms that
//   need to calculate properties that involve all walkers, e.g. pure estimators    
class TaskGroup {

  public:

  TaskGroup(std::string name):tgname(name),initialized(false),
     verbose(true)
  {}
  ~TaskGroup() {};

  void setBuffer(SMDenseVector<ComplexType>* buf) { commBuff = buf; }

  SMDenseVector<ComplexType>* getBuffer() { return commBuff; }

  bool setup(int ncore=0, int nnode=0, bool print=false) { 
 
    verbose = print;

    MPI_Comm_rank(MPI_COMM_WORLD,&global_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&global_nproc);
   
    app_log()<<std::endl
             <<"**************************************************************\n"
             <<" Setting up Task Group: " <<tgname <<std::endl; 


    MPI_Info info;
    // MPI_COMM_NODE_LOCAL is the comm local to a node, 
    // including all cores that can share a shared memory window
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&MPI_COMM_NODE_LOCAL);

    MPI_Comm_rank(MPI_COMM_NODE_LOCAL,&core_number);
    MPI_Comm_size(MPI_COMM_NODE_LOCAL,&tot_cores);    

    // split World along coreid
    MPI_Comm_split(MPI_COMM_WORLD,core_number,global_rank,&MPI_COMM_HEAD_OF_NODES);

    MPI_Comm_rank(MPI_COMM_HEAD_OF_NODES,&node_number);
    MPI_Comm_size(MPI_COMM_HEAD_OF_NODES,&tot_nodes);

    ncores_per_TG = (ncore==0)?tot_cores:ncore;
    nnodes_per_TG = (nnode==0)?tot_nodes:nnode;

    // check for consistency
    int dum = tot_cores;
    MPI_Bcast(&dum,1,MPI_INT,0,MPI_COMM_WORLD);
    if(dum != tot_cores) {
      app_error()<<" Error: Inconsistent number of cores in Task Group: " <<dum <<" " 
                 <<tot_cores <<" " <<global_rank <<std::endl;
      return false;
    }
    dum = tot_nodes;
    MPI_Bcast(&dum,1,MPI_INT,0,MPI_COMM_WORLD);
    if(dum != tot_nodes) {
      app_error()<<" Error: Inconsistent number of nodes: " <<dum <<" " <<tot_nodes 
                 <<" " <<global_rank <<std::endl;
      return false;
    }
    dum = node_number;
    MPI_Bcast(&dum,1,MPI_INT,0,MPI_COMM_NODE_LOCAL);
    if(dum != node_number) {
      app_error()<<" Error: Inconsistent node number: " <<dum <<" " <<node_number
                 <<" " <<global_rank <<std::endl;
      return false;
    }    

    if( tot_cores%ncores_per_TG != 0  ) {
      app_error()<<"Found " <<tot_cores <<" cores per node. " <<std::endl;
      app_error()<<" Error in TaskGroup setup(): Number of cores per node is not divisible by requested number of cores in Task Group." <<std::endl;
      return false;
    }
    if( tot_nodes%nnodes_per_TG != 0  ) {
      app_error()<<"Found " <<tot_nodes <<" nodes. " <<std::endl;
      app_error()<<" Error in TaskGroup setup(): Number of nodes is not divisible by requested number of nodes in Task Group." <<std::endl;
      return false;
    }
       
    // setup TG grid 
    nrows = tot_cores/ncores_per_TG;
    ncols = tot_nodes/nnodes_per_TG; 
    mycol = node_number/nnodes_per_TG;     // simple square asignment 
    node_in_TG = node_number%nnodes_per_TG; 
    myrow = core_number/ncores_per_TG; 
    TG_number = mycol + ncols*myrow; 
    number_of_TGs = nrows*ncols;

    // split communicator
    MPI_Comm_split(MPI_COMM_WORLD,TG_number,global_rank,&MPI_COMM_TG);
    MPI_Comm_rank(MPI_COMM_TG,&TG_rank);
    MPI_Comm_size(MPI_COMM_TG,&TG_nproc);

    if(nnodes_per_TG > 1) { 
      MPI_Comm_split(MPI_COMM_TG,node_in_TG,global_rank,&MPI_COMM_TG_LOCAL);
    } else {
      MPI_COMM_TG_LOCAL = MPI_COMM_TG;
    }

    MPI_Comm_rank(MPI_COMM_TG_LOCAL,&core_rank);    
    MPI_Comm_split(MPI_COMM_TG,core_rank,global_rank,&MPI_COMM_TG_HEADS);

    TG_root = false;
    if(TG_rank==0) TG_root = true;
    core_root = (core_rank==0);

    if( TG_rank != node_in_TG*ncores_per_TG+core_rank ) {
      app_error()<<" Error in TG::setup(): Unexpected TG_rank: " <<TG_rank <<" " <<node_in_TG <<" " <<core_rank <<" " <<node_in_TG*ncores_per_TG+core_rank <<std::endl;
      APP_ABORT("Error in TG::setup(): Unexpected TG_rank \n");   
    }    

    // define ring communication pattern
    // these are the ranks (in TGcomm) of cores with the same core_rank 
    // on nodes with id +-1 (with respect to local node). 
    next_core_circular_node_pattern = ((node_in_TG+1)%nnodes_per_TG)*ncores_per_TG+core_rank; 
    if(node_in_TG==0)
      prev_core_circular_node_pattern = (nnodes_per_TG-1)*ncores_per_TG+core_rank;   
    else
      prev_core_circular_node_pattern = ((node_in_TG-1)%nnodes_per_TG)*ncores_per_TG+core_rank;   

    app_log()<<"**************************************************************" <<std::endl;
    initialized=true;
    return true;
  }

  void set_min_max(int min, int max) {
    min_index=min;
    max_index=max;
  }

  void get_min_max(int& min, int& max) const {
    min=min_index;
    max=max_index;
  }

  // over full TG using mpi communicator 
  void barrier() {
    MPI_Barrier(MPI_COMM_TG);
  }

  // over local node using boost sync 
  void local_barrier() {
    MPI_Barrier(MPI_COMM_TG_LOCAL);
  }

  MPI_Comm getTGComm() const { return MPI_COMM_TG; }

  MPI_Comm getTGCommHeads() const { return MPI_COMM_TG_HEADS; }

  MPI_Comm getTGCommLocal() const { return MPI_COMM_TG_LOCAL; }

  MPI_Comm getNodeCommLocal() const { return MPI_COMM_NODE_LOCAL; }

  MPI_Comm getHeadOfNodesComm() const { return MPI_COMM_HEAD_OF_NODES;}

  void allgather_TG(std::vector<int>& l, std::vector<int>& g) {
    MPI_Allgather(l.data(),l.size(),MPI_INT,
                  g.data(),l.size(),MPI_INT, MPI_COMM_TG);
  }  

  // size is in units of ComplexType and represents (walker_size)*(number_of_walkers) 
  void resize_buffer(int& size) 
  {
    int sz=size;
    MPI_Allreduce(&sz,&size,1,MPI_INT,MPI_MAX,MPI_COMM_TG);
    // reset SM is necessary 
    commBuff->resize(size);
  }

  int getGlobalRank() const { return global_rank; }

  int getTotalNodes() const { return tot_nodes; }

  int getTotalCores() const { return tot_cores; }

  int getNodeID() const { return node_number; }

  int getCoreID() const { return core_number; }

  int getCoreRank() const { return core_rank; }

  int getLocalNodeNumber() const { return node_in_TG; }

  int getTGNumber() const { return TG_number; }

  int getNumberOfTGs() const { return number_of_TGs; }

  int getTGRank() const { return TG_rank; }

  int getTGSize() const { return TG_nproc; }

  int getNCoresPerTG() const { return ncores_per_TG; }

  int getNNodesPerTG() const { return nnodes_per_TG; }

  int getNextRingPattern() const { return next_core_circular_node_pattern; } 

  int getPrevRingPattern() const { return prev_core_circular_node_pattern; } 
 
  void getSetupInfo(std::vector<int>& data) const
  {  
    data.resize(5);
    data[0]=node_number;
    data[1]=core_number;
    data[2]=tot_nodes;
    data[3]=tot_cores;
    data[4]=ncores_per_TG; 
  }
 
  // must be setup externally to be able to reuse between different TG 
  SMDenseVector<ComplexType>* commBuff;  

  std::string tgname;

  bool verbose;
  bool initialized;
  
  int global_rank;
  int global_nproc;
  int node_number, core_number, tot_nodes, tot_cores;
  int TG_number; 
  int number_of_TGs; 
  // TGs are defined in a 2-D framwork. Rows correspond to different groups in a node 
  // Cols correspond to division of nodes into groups. Every MPI task belongs to a specific
  // TG given by the myrow and mycol integer.      
  int myrow, mycol, nrows, ncols, node_in_TG;
  bool TG_root;            // over full TG
  int TG_rank, TG_nproc;   // over full TG, notice that nproc = ncores_per_TG * nnodes_per_TG 
  bool core_root;          // over local node
  int core_rank;           // over local node 
  int next_core_circular_node_pattern, prev_core_circular_node_pattern; 
  MPI_Comm MPI_COMM_TG;   // Communicator over all cores in a given TG 
  MPI_Comm MPI_COMM_TG_HEADS;   // Communicator over all cores in a given TG 
  MPI_Comm MPI_COMM_TG_LOCAL;   // Communicator over all cores in a given TG that reside in the given node 
  MPI_Comm MPI_COMM_NODE_LOCAL; // Communicator over all cores of a node. 
  MPI_Comm MPI_COMM_HEAD_OF_NODES;  // deceiving name for historical reasons, this is a split of COMM_WORLD over core_number. 
  std::vector<SPComplexType> local_buffer;

  int ncores_per_TG;  // total number of cores in all nodes must be a multiple 
  int nnodes_per_TG;  // total number of nodes in communicator must be a multiple  
  int min_index, max_index;  

};

}


#endif
