


#include <glib.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include "leveldb/db.h"

using leveldb;

#define UINT unsigned int
#define BOOL unsigned short

#define EKT_NODE struct ExtraKMeansTreeNode
#define EKT_NODE_P struct ExtraKMeansTreeNode*
#define EKT_ARRAY struct EKTArray
#define EKT_ARRAY_P struct EKTArray* 

struct ExtraKMeansTreeNode
{
    ExtraKMeansTreeNode_P parent;
    ExtraKMeansTreeNode_P child_node_list;
    BOOL is_leaf;
    UINT spliter_row_index;
    UINT* cluster_data_index_arr;
};

struct EKTArray
{
    UINT* array;
    UINT length;
};


void load_data(){
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, "./db/testdb", &db);
    assert(status.ok());

    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
}

void function(EKT_ARRAY_P sparse_data_arr, EKT_ARRAY_P sparse_data_index_arr, ){

    UINT n_user = sparse_data_index_arr->length - 1;
    GHashTable* hash = g_hash_table_new(g_direct_hash, g_direct_equal);

}


int main(int argc, char const *argv[])
{
    
    return 0;
}

/* compile cmd
gcc disk_spider_dirtree.c -o disk_spider_dirtree_c `pkg-config --cflags --libs glib-2.0`
*/