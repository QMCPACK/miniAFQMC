//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign   
//		      Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign 
//////////////////////////////////////////////////////////////////////////////////////


#include <Configuration.h>
#include <io/hdf_archive.h>
namespace qmcplusplus
{
//const hid_t hdf_archive::is_closed;
hdf_archive::hdf_archive()
  : file_id(is_closed), access_id(H5P_DEFAULT), xfer_plist(H5P_DEFAULT)
{
  H5Eget_auto (&err_func, &client_data);
  H5Eset_auto (NULL, NULL);
  set_access_plist();
}

hdf_archive::~hdf_archive()
{
#if defined(H5_HAVE_PARALLEL) && defined(ENABLE_PHDF5)
  if(xfer_plist != H5P_DEFAULT) H5Pclose(xfer_plist);
  if(access_id != H5P_DEFAULT) H5Pclose(access_id);
#endif
  close();
  H5Eset_auto (err_func, client_data);
}

void hdf_archive::close()
{

  while(!group_id.empty())
  {
    hid_t gid=group_id.top();
    group_id.pop();
    H5Gclose(gid);
  }
  if(file_id!=is_closed)
    H5Fclose(file_id);
  file_id=is_closed;
}

void hdf_archive::set_access_plist()
{
  access_id=H5P_DEFAULT;
    Mode.set(IS_PARALLEL,false);
    Mode.set(NOIO,false);
}

bool hdf_archive::create(const std::string& fname, unsigned flags)
{
  //not I/O node, do nothing
  if(Mode[NOIO]) return true;
  close(); 
  file_id = H5Fcreate(fname.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,access_id);
  return file_id != is_closed;
}

bool hdf_archive::open(const std::string& fname,unsigned flags)
{
  if(Mode[NOIO])
    return true;
  close();
  file_id = H5Fopen(fname.c_str(),flags,access_id);
  return file_id != is_closed;
}

bool hdf_archive::is_group(const std::string& aname)
{
  if(Mode[NOIO])
    return true;
  if(file_id==is_closed)
    return false;
  hid_t p=group_id.empty()? file_id:group_id.top();
  p=(aname[0]=='/')?file_id:p;
  hid_t g=H5Gopen(p,aname.c_str());
  if(g<0)
    return false;
  H5Gclose(g);
  return true;
}

hid_t hdf_archive::push(const std::string& gname, bool createit)
{
  if(Mode[NOIO]||file_id==is_closed)
    return is_closed;
  hid_t p=group_id.empty()? file_id:group_id.top();
  hid_t g=H5Gopen(p,gname.c_str());
  if(g<0 && createit)
  {
    g= H5Gcreate(p,gname.c_str(),0);
  }
  if(g!=is_closed)
    group_id.push(g);
  return g;
}

}
