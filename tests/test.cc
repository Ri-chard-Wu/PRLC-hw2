

#include <cassert>
#include <iostream>
#include <unordered_map>
using namespace std;
  




template <class Ket_t>
class MapQueue{
    public:

    MapQueue(){
        nNodes = 0;
    }



    bool try_pop_front(Ket_t* key){
        if(!nNodes) return false;

        *key = frontKey;
        remove(frontKey);

        return true;
    }



    bool try_pop_back(Ket_t* key){
        if(!nNodes) return false;

        *key = backKey;
        remove(backKey);

        return true;
    }


    void push(Ket_t key){
        // - to update:
        // 1. nNodes
        // 2. nodeTbl (neighboring nodes, and insert the new node)
        // 3. backKey, frontKey


        if(is_inside(key)){
            fprintf(stderr, "[MapQueue::remove()] key already exist: %d\n", key);
            return;
        }


        Node node;

        if(!nNodes){
            node.nextKey = key;
            node.prevKey = key;
            nodeTbl[key] = node;

            frontKey = key;
            backKey = key;
            
            nNodes++;
        }
        else{
            node.nextKey = frontKey;
            node.prevKey = backKey;
            nodeTbl[key] = node;

            nodeTbl[backKey].nextKey = key;
            nodeTbl[frontKey].prevKey = key;

            backKey = key;

            nNodes++;
        }
    }


    bool is_inside(Ket_t key){
        return !(nodeTbl.find(key) == nodeTbl.end());
    }


    void remove(Ket_t key){

        if(!is_inside(key)){
            fprintf(stderr, "[MapQueue::remove()] key not found: %d\n", key);
            return;
        }




        if((key != frontKey) && (key != backKey)){

            Ket_t nextkey, prevKey;

            nextkey = nodeTbl[key].nextKey;
            prevKey = nodeTbl[key].prevKey;

            nodeTbl[nextkey].prevKey = prevKey;
            nodeTbl[prevKey].nextKey = nextkey;

            nodeTbl.erase(key);
            nNodes--;

        }
        else if((key == frontKey) && (key == backKey)){

            nodeTbl.erase(key);
            nNodes--;

            assert(nNodes == 0);
        }
        else if(key == frontKey){

            Ket_t nextkey, prevKey;
            nextkey = nodeTbl[frontKey].nextKey;
            prevKey = nodeTbl[frontKey].prevKey;

            nodeTbl[nextkey].prevKey = prevKey;
            nodeTbl[prevKey].nextKey = nextkey;

            frontKey = nextkey;

            nodeTbl.erase(key);
            nNodes--;

        }
        else if(key == backKey){
            // backKey = prevKey;

            Ket_t nextkey, prevKey;
            nextkey = nodeTbl[backKey].nextKey;
            prevKey = nodeTbl[backKey].prevKey;

            nodeTbl[nextkey].prevKey = prevKey;
            nodeTbl[prevKey].nextKey = nextkey;

            backKey = prevKey;

            nodeTbl.erase(key);
            nNodes--;
        }


    }



    struct Node{
        Ket_t nextKey;
        Ket_t prevKey;
    };



    unordered_map<Ket_t, Node> nodeTbl;
    Ket_t frontKey, backKey;
    int nNodes;
};



int main()
{


    MapQueue<int> mq;

    mq.push(21);
    mq.push(122);
    mq.push(102); 
    mq.push(100);
    mq.remove(100);
    mq.push(1202); 
    mq.push(430);
    mq.push(4300);    

    int out;
    for(int i=0;i<3;i++){
        if(mq.try_pop_front(&out)){
            cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
        }
    }
    cout<<endl;

    mq.remove(1202);
    mq.push(12020); 
    mq.push(4310);
    mq.push(43000);   

    for(int i=0;i<2;i++){
        if(mq.try_pop_front(&out)){
            cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
        }  
    }
    cout<<endl;


    for(int i=0;i<2;i++){
        if(mq.try_pop_back(&out)){
            cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
        }  
    }
    cout<<endl;
    // mq.remove(12020);
    mq.push(2100); 
    mq.push(20100); 

    for(int i=0;i<210;i++){
        if(mq.try_pop_back(&out)){
            cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
        }  
    }
    cout<<endl;


    // - remove the only element

    // MapQueue<int> mq;

    // mq.push(21);
    // mq.remove(21);
    // mq.push(122);
    // mq.push(102); 
    // mq.push(100);


    // int out;
    // for(int i=0;i<3;i++){
    //     if(mq.try_pop(&out)){
    //         cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
    //     }
        
    // }













    // mq.push(211);
    // mq.push(31);
    // mq.push(21);

    // // int n = mq.nNodes;
    // int out;
    // for(int i=0;i<2;i++){
    //     if(mq.try_pop(&out)){
    //         cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
    //     }
        
    // }
    // cout<<endl;


    // mq.push(10);
    // mq.push(56);
    // mq.push(12);
    // mq.push(120);
    // mq.push(1200);

    // for(int i=0;i<3;i++){
    //     if(mq.try_pop(&out)){
    //         cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
    //     }
        
    // }
    // cout<<endl;

    // mq.remove(31);
    // mq.remove(120);
    // mq.push(30);

    // if(!mq.is_inside(31)){
    //     cout<<"not inside"<<endl;
    // }

    // if(mq.is_inside(1200)){
    //     cout<<"is inside"<<endl;
    // }

    // // mq.remove(10);
    // // mq.try_pop(&out);
    // mq.push(1000);

    // int n = mq.nNodes;
    // for(int i=0;i<n;i++){
    //     if(mq.try_pop(&out)){
    //         cout<<"out: "<<out<<", nNodes:"<<mq.nNodes<<endl;
    //     }
        
    // }
    // cout<<endl;





//   unordered_map<string, int> umap;
  
//   // inserting values by using [] operator
//   umap["GeeksforGeeks"] = 10;
//   umap["Practice"] = 20;
//   umap["Contribute"] = 30;
//   umap["Contribute"]++;
//   umap.erase("GeeksforGeeks");
//   umap.erase("GeeksforGeeks");
//   // Traversing an unordered map
//   for (auto x : umap)
//     cout << x.first << " " << 
//             x.second << endl;
}