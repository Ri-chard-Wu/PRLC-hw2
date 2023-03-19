#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lodepng.h>

#define GLM_FORCE_SWIZZLE  
#include <glm/glm.hpp>

#include <mpi.h>
#include<pthread.h>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_queue.h"
#include <unordered_map>
#include <queue>

using namespace std;
using namespace oneapi::tbb;

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2;  
typedef glm::dvec3 vec3;  
typedef glm::dvec4 vec4;  
typedef glm::dmat3 mat3;  

int AA = 2; 

double power = 8.0;           
double md_iter = 24;          
double ray_step = 10000;      
double shadow_step = 1500;    
double step_limiter = 0.2;    
double ray_multiplier = 0.1;  
double bailout = 2.0;         
double eps = 0.0005;          
double FOV = 1.5;             
double far_plane = 100.;      
 


void write_png(const char* filename, unsigned char* raw_image, unsigned int width, unsigned int height) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);

    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}



double md(vec3 p, double& trap) {
    vec3 v = p;
    double dr = 1.;            
    double r = glm::length(v); 
    trap = r;

    for (int i = 0; i < md_iter; ++i) {
        double theta = glm::atan(v.y, v.x) * power;
        double phi = glm::asin(v.z / r) * power;
        dr = power * glm::pow(r, power - 1.) * dr + 1.;
        v = p + glm::pow(r, power) *
                    vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));  // update vk+1

        trap = glm::min(trap, r);

        r = glm::length(v);      
        if (r > bailout) break;  
    }
    return 0.5 * log(r) * r / dr;  

}


double map(vec3 p, double& trap, int& ID) {
    vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));

    vec3 rp = mat3(1.,   0.,    0.,
                   0., rt.x, -rt.y, 
                   0., rt.y,  rt.x) * p;  
    ID = 1;
    return md(rp, trap);
}


double map(vec3 p) {
    double dmy;  // dummy
    int dmy2;    // dummy2
    return map(p, dmy, dmy2);
}



vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2. * pi * (c * t + d));
}


double softshadow(vec3 ro, vec3 rd, double k) {
    double res = 1.0;
    double t = 0.;  
    for (int i = 0; i < shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = glm::min(res, k * h / t);  
        if (res < 0.02) return 0.02;
        t += glm::clamp(h, .001, step_limiter);  // move ray
    }
    return glm::clamp(res, .02, 1.);
}


vec3 calcNor(vec3 p) {
    vec2 e = vec2(eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),  // dx
                          map(p + e.yxy()) - map(p - e.yxy()),  // dy
                          map(p + e.yyx()) - map(p - e.yyx())   // dz
                    ));
}


double trace(vec3 ro, vec3 rd, double& trap, int& ID) {
    double t = 0;    
    double len = 0;  

    for (int i = 0; i < ray_step; ++i) {

        len = map(ro + rd * t, trap, ID);  
        if (glm::abs(len) < eps || t > far_plane) break;
        t += len * ray_multiplier;
    }

    return t < far_plane ? t : -1.;  
}









#define MPI_TAG_JOB_ASSIGN 0
#define MPI_TAG_JOB_CANCEL 1
#define MPI_TAG_TERMINATE 2
#define MPI_TAG_JOB_UPLOAD 3

#define BUF_SIZE_JOB_UPLOAD 10000
#define BUF_SIZE_JOB_SCHDL 5000

typedef int jobIdx_t;



template <class Ket_t>
class MapQueue{
    public:

    MapQueue(){
        nNodes = 0;
    }

    bool empty(){
        return nNodes == 0;
    }

    int size(){
        return nNodes;
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


    void push_back(Ket_t key){
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


    bool remove(Ket_t key){

        if(!is_inside(key)){
            // fprintf(stderr, "[MapQueue::remove()] key not found: %d\n", key);
            return false;
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

            Ket_t nextkey, prevKey;
            nextkey = nodeTbl[backKey].nextKey;
            prevKey = nodeTbl[backKey].prevKey;

            nodeTbl[nextkey].prevKey = prevKey;
            nodeTbl[prevKey].nextKey = nextkey;

            backKey = prevKey;

            nodeTbl.erase(key);
            nNodes--;
        }

        return true;
    }

    struct Node{
        Ket_t nextKey;
        Ket_t prevKey;
    };

    unordered_map<Ket_t, Node> nodeTbl;
    Ket_t frontKey, backKey;
    int nNodes;
};









struct Job{
    jobIdx_t idx;     
    vec4 result;
};


struct JobNode{
    jobIdx_t jobIdx;
    JobNode* next;
    JobNode* prev;
};


struct PCB{
    // int nJobs;
    // JobNode* jobQueueFront;
    // JobNode* jobQueueBack;
    MapQueue<int> jobQueue;
};








class ThreadManager{
    public:

    ThreadManager(char** argv, int rank, int worldSize){

        this->rank = rank;
        this->worldSize = worldSize;

        num_threads = atoi(argv[1]);
        camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
        target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
        width = atoi(argv[8]);
        height = atoi(argv[9]);
        nTotalJobs = width * height;
        nThrds = num_threads - 1;
    }


    void start_thread(){

        
        thrds = new pthread_t[nThrds];
        entryArg* argPtr;
        int ret;

        for(int i = 0; i < nThrds; i++){
            argPtr = new entryArg;
            argPtr->objPtr = this;
            argPtr->tid = i;

            ret = pthread_create(&thrds[i], NULL, thread_entry, (void *)argPtr);
            if(ret != 0) printf("create thread failed.\n");
        }
    }


    static void * thread_entry(void *arg) {
        entryArg *argPtr = (entryArg *)arg;

        int tid = argPtr->tid;
        ThreadManager *objPtr = (ThreadManager *)argPtr->objPtr;
        objPtr->job(tid);
        
        return NULL;
    }


    void join_thread()
    {
        for (int i = 0; i < nThrds; i++){
            pthread_join(thrds[i], NULL);
        }  

        fprintf(stdout, "[proc %d][join_thread()] all threads joined.\n", rank);
    }





    void job(int tid){

        Job job;

        while(1){
            if(!jobQueue.empty()){

                if(!jobQueue.try_pop(job)){
                    fprintf(stderr, "[pid %d][job()] pop failed.\n", rank);
                    continue;
                }

                int i, j;
            
                jobIdx2ImgCoord(job.idx, &i, &j);

                job.result = vec4(0.);

                for (int m = 0; m < AA; ++m) {
                    for (int n = 0; n < AA; ++n) {
                        
                        job.result += partial_AA(i, j, m, n);
                    }
                }       
 
                job.result /= (double)(AA * AA);
                job.result *= 255.0;

                jobDoneQueue.push(job);

                // fprintf(stderr, "[pid %d][tid %d] job.idx: %d \n", rank, tid, job.idx);
            }
        }
    }



    void one_job(){

        Job job;
    
        if(!jobQueue.empty()){
            if(!jobQueue.try_pop(job)){
                fprintf(stderr, "[pid %d][one_job()] pop failed.\n", rank);
                return;
            }
            
            int i, j;
        
            jobIdx2ImgCoord(job.idx, &i, &j);

            job.result = vec4(0.);

            for (int m = 0; m < AA; ++m) {
                for (int n = 0; n < AA; ++n) {
                    
                    job.result += partial_AA(i, j, m, n);
                }
            } 
            
            job.result /= (double)(AA * AA);
            job.result *= 255.0;

            
    
            jobDoneQueue.push(job);
        }
        
    }


    void enqueue_job(jobIdx_t jobIdx){

        Job job;
        job.idx = jobIdx;

        jobQueue.push(job);
    }


    void jobIdx2ImgCoord(int idx, int* i, int* j){
        *i = idx / width;
        *j = idx % width;        
    }

    void imgCoord2JobIdx(int i, int j, int* idx){
        *idx = i * ((int)width) + j;      
    }


    vec4 partial_AA(int i, int j, int m, int n){

        vec2 iResolution = vec2(width, height);

        vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

        vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
        uv.y *= -1;  

        vec3 ro = camera_pos;               
        vec3 ta = target_pos;               
        vec3 cf = glm::normalize(ta - ro);  
        vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.))); 
        vec3 cu = glm::normalize(glm::cross(cs, cf));               
        vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf); 
        
        double trap;  
        int objID;    
        double d = trace(ro, rd, trap, objID); // dependent loop, 10000 * 24
        
        vec3 col(0.);                          
        vec3 sd = glm::normalize(camera_pos);  
        vec3 sc = vec3(1., .9, .717);          
        

        if (d < 0.) {        
            col = vec3(0.);  
        } else {
            vec3 pos = ro + rd * d;             
            vec3 nr = calcNor(pos);             
            vec3 hal = glm::normalize(sd - rd); 
            

            col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.), vec3(.0, .1, .2));  
            vec3 ambc = vec3(0.3); 
            double gloss = 32.;    

            double amb =
                (0.7 + 0.3 * nr.y) *
                (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));  

            double sdw = softshadow(pos + .001 * nr, sd, 16.);    // dependent loop, 1500
                
            double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;   
            double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) * dif;  


            vec3 lin(0.);
            lin += ambc * (.05 + .95 * amb);  
            lin += sc * dif * 0.8;            
            col *= lin;

            col = glm::pow(col, vec3(.7, .9, 1.)); 
            col += spe * 0.8;                      
        }
        

        col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.); 

        return vec4(col, 1.);
    }


    int rank, worldSize;

    concurrent_queue<Job> jobQueue;
    concurrent_queue<Job> jobDoneQueue;


    unsigned int nThrds;
    pthread_t *thrds;
    

    unsigned int num_threads;  
    unsigned int width;        
    unsigned int height;  
    vec3 camera_pos; 
    vec3 target_pos; 
    unsigned int nTotalJobs;


    struct entryArg{
        ThreadManager* objPtr;
        int tid;
    };

};












class ProcessManager{ 
    public:
    
    ProcessManager(char** argv, int rank, int worldSize){
        
        tmPtr = new ThreadManager(argv, rank, worldSize);
        filename = argv[10];
    
    
        this->rank = rank;
        this->worldSize = worldSize;
        nStaticJobs =  tmPtr->nTotalJobs / worldSize; 
        startIdx = nStaticJobs * rank;
        if(rank < worldSize - 1){stopIdx = nStaticJobs * (rank + 1) - 1;}
        else if(rank == worldSize - 1){stopIdx = tmPtr->nTotalJobs - 1;}

        recvCnt = new int[worldSize];
        for(int pid=1; pid<worldSize; pid++){
            recvCnt[pid] = 0;
        }
        nJobsCmpltd = 0;



        procCtrlTbl = new PCB[worldSize];
        // for(int i=0;i<worldSize;i++){
        //     procCtrlTbl[i].jobQueueFront = NULL;
        //     procCtrlTbl[i].jobQueueBack = NULL;
        //     procCtrlTbl[i].nJobs = 0;
        // }


        raw_image = new unsigned char[tmPtr->width * tmPtr->height * 4];
        image = new unsigned char*[tmPtr->height];

        for (int i = 0; i < tmPtr->height; ++i) {
            image[i] = raw_image + i * tmPtr->width * 4;
        }


        for(int i=startIdx; i<=stopIdx; i++){
            dynamicJobQueue.push_back(i);
        }

        init_procCtrlTbl();
        
    }


    void init_procCtrlTbl(){
        int s, e;

        for(int pid=0; pid < worldSize; pid++){
            
            s = nStaticJobs * pid;
            if(pid < worldSize - 1){e = nStaticJobs * (pid + 1) - 1;}
            else if(pid == worldSize - 1){e = tmPtr->nTotalJobs - 1;}
            else{
                fprintf(stderr, "[proc 0][init_procCtrlTbl()] Invalid pid: %d\n", pid);
                exit(1);                
            }
            
            for(int jobIdx = s; jobIdx <= e; jobIdx++){
                pcb_push_back_job(pid, jobIdx);
            }

        }   

        fprintf(stderr, "[proc 0][init_procCtrlTbl()] after init, proc njob:\n", rank);
        print_proc_nJob();

    }
    


    void dynamic_job_enqueue(){
        if(dynamicJobQueue.empty()) return;

        static int cnt=startIdx;

        if(tmPtr->jobQueue.unsafe_size() < 10 * tmPtr->nThrds){
            for(int i=0; i < 10 * tmPtr->nThrds; i++){
                
                
         
                jobIdx_t jobIdx;
                if(dynamicJobQueue.try_pop_front(&jobIdx)){
                    Job job;
                    job.idx = jobIdx;

                    assert(jobIdx == cnt);
                    assert(cnt <= stopIdx);
                    cnt++;
                    // fprintf(stderr, "[proc 0] jobIdx: %d, cnt: %d\n", jobIdx, cnt);

                    tmPtr->jobQueue.push(job);
                }
            }
        }
    }


    void start_process(){

        dynamic_job_enqueue();
        tmPtr->start_thread();

        while(1){

            if(!tmPtr->jobQueue.empty()){

                tmPtr->one_job(); 

                if(tmPtr->jobDoneQueue.unsafe_size() > 2500){
                    local_receive_completed_jobs();
                }                
            }
            else{
                local_receive_completed_jobs();
                if(check_send_terminate_signal()) return;
            }

            receive_completed_jobs();
            // dynamic_job_schedule();

            dynamic_job_enqueue();
        }  
    }




    void local_receive_completed_jobs(){

        Job job;
        int ofst = 0, i, j; 

        while(!tmPtr->jobDoneQueue.empty()){
            if(!tmPtr->jobDoneQueue.try_pop(job)){
                fprintf(stderr, "[local_receive_completed_jobs()] pop failed.\n");
                return;
            }

            // if(!pcb_remove_job(0, job.idx)) ;

            tmPtr->jobIdx2ImgCoord(job.idx, &i, &j);

            if(!pcb_remove_job(0, job.idx)){

                fprintf(stderr, "[proc %d] Already in image[]: jobIdx: %d, i: %d, j: %d, data: (%d, %d, %d).\n", 
            rank, job.idx, i, j, image[i][4 * j + 0], image[i][4 * j + 1], image[i][4 * j + 2]); 
                
                fprintf(stderr, "[proc %d] duplicate done data: jobIdx: %d, i: %d, j: %d, data: (%d, %d, %d).\n", 
            rank, job.idx, i, j, (int)job.result.r, (int)job.result.g, (int)job.result.b); 
                continue;
            }
            
            image[i][4 * j + 0] = (unsigned char)job.result.r;  
            image[i][4 * j + 1] = (unsigned char)job.result.g;  
            image[i][4 * j + 2] = (unsigned char)job.result.b;  
            image[i][4 * j + 3] = (unsigned char)255;  

            ofst++;
        }

        
        if(ofst > 0){
            nJobsCmpltd += ofst;

            fprintf(stdout, "[proc %d][local_receive_completed_jobs()] from pid 0 recvSz: %d\n",
            rank, nJobsCmpltd);                                     
        }
    }



    
    void receive_completed_jobs(){

        int flag, recvSz, i, j, data, jobIdx;
        int dataMask = (1 << 8) - 1;
        PCB* pcbPtr;
        MPI_Status status;
        
        
        for(int pid=1; pid<worldSize; pid++){


            MPI_Iprobe(pid, MPI_TAG_JOB_UPLOAD, MPI_COMM_WORLD, &flag, &status);
            if(!flag){continue;}        

            MPI_Get_count(&status, MPI_INT, &recvSz);
            MPI_Recv(jobDoneBuf, recvSz, MPI_INT, pid, MPI_TAG_JOB_UPLOAD,
                                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            recvCnt[pid] += recvSz;
            
            fprintf(stdout, "[proc %d][receive_completed_jobs()] from pid %d recvSz: %d\n",
            rank, pid, recvCnt[pid]/2); 

         
            for(int k = 0; k < recvSz; k += 2){

                jobIdx = jobDoneBuf[k];

                tmPtr->jobIdx2ImgCoord(jobIdx, &i, &j);
                data = jobDoneBuf[k+1];
               
                if(!pcb_remove_job(pid, jobIdx)){

                    fprintf(stderr, "[proc %d] pid %d, Already in image[]: jobIdx: %d, i: %d, j: %d, data: (%d, %d, %d).\n", 
              rank, pid, jobIdx, i, j, image[i][4 * j + 0], image[i][4 * j + 1], image[i][4 * j + 2]); 
                    
                    fprintf(stderr, "[proc %d] pid %d, duplicate done data: jobIdx: %d, i: %d, j: %d, data: (%d, %d, %d).\n", 
              rank, pid, jobIdx, i, j, (data >> 24) & dataMask, ((data >> 16) & dataMask), ((data >> 8 ) & dataMask)); 
                    continue;
                }
        //         fprintf(stderr, "[proc %d][receive_completed_jobs()] jobIdx: %d, i: %d, j: %d, data: (%d, %d, %d).\n", 
        //   rank, jobIdx, i, j, (data >> 24) & dataMask, ((data >> 16) & dataMask), ((data >> 8 ) & dataMask)); 

                image[i][4 * j + 0] = (unsigned char)((data >> 24) & dataMask);  
                image[i][4 * j + 1] = (unsigned char)((data >> 16) & dataMask);  
                image[i][4 * j + 2] = (unsigned char)((data >> 8 ) & dataMask);  
                image[i][4 * j + 3] = (unsigned char)255;  


                if(jobIdx < 0 || jobIdx >= (tmPtr->nTotalJobs)){
                    fprintf(stderr, "[proc %d][receive_completed_jobs()] job idx out of range.\n", rank);            
                    exit(1);
                }

                // pcb_remove_job(pid, jobIdx);

            }
        }
    }


    bool pcb_pop_front_job(int pid, int* jobIdx){
        return procCtrlTbl[pid].jobQueue.try_pop_front(jobIdx);
    }


    bool pcb_pop_back_job(int pid, int* jobIdx){
        return procCtrlTbl[pid].jobQueue.try_pop_back(jobIdx);
    }


    void pcb_push_back_job(int pid, jobIdx_t jobIdx){
        procCtrlTbl[pid].jobQueue.push_back(jobIdx);

    }


    bool pcb_remove_job(int pid, jobIdx_t jobIdx){
        
        if(!procCtrlTbl[pid].jobQueue.remove(jobIdx)){
            fprintf(stderr, "[pid %d][pcb_remove_job()] jobIdx not found: %d\n", pid, jobIdx);
            return false;
        }
        return true;
        
    }

    

    void dynamic_job_schedule(){

        PCB* pcbPtr;
        int minNJob = 1 << 30, minNJobPid;
        int maxNJob = 0, maxNJobPid;

        for(int pid=0; pid<worldSize; pid++){

            if(minNJob > procCtrlTbl[pid].jobQueue.size()){
                minNJob = procCtrlTbl[pid].jobQueue.size();
                minNJobPid = pid;
            }

            if(maxNJob < procCtrlTbl[pid].jobQueue.size()){
                maxNJob = procCtrlTbl[pid].jobQueue.size();
                maxNJobPid = pid;
            }
        }


        int jobIdx;
        if((maxNJob - minNJob >= 2 *  BUF_SIZE_JOB_SCHDL) && (minNJob > BUF_SIZE_JOB_SCHDL)){
            
            fprintf(stderr, "[dynamic_job_schedule()] maxNJob: %d, maxNJobPid: %d, minNJob: %d, minNJobPid: %d\n",
                                maxNJob, maxNJobPid, minNJob, minNJobPid);  
            
            for(int i=0; i < BUF_SIZE_JOB_SCHDL; i++){
                
                if(!pcb_pop_back_job(maxNJobPid, &jobIdx)){break;}
                pcb_push_back_job(minNJobPid, jobIdx);
                
                jobSchdlBuf[i] = jobIdx;
            }

            dynamic_assign_job(minNJobPid, jobSchdlBuf, BUF_SIZE_JOB_SCHDL);
            dynamic_cancel_job(maxNJobPid, jobSchdlBuf, BUF_SIZE_JOB_SCHDL);
        }
    }

    

    void dynamic_assign_job(int pid, int* buf, int size){
        if(pid != 0){
            MPI_Send(buf, size, MPI_INT, pid, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD);
        }else{
            for(int i=0; i<size; i++){
                dynamicJobQueue.push_back(buf[i]);
            }            
        }                 
    }


    void dynamic_cancel_job(int pid, int* buf, int size){
               
        if(pid != 0){
            MPI_Send(buf, size, MPI_INT, pid, MPI_TAG_JOB_CANCEL, MPI_COMM_WORLD); 
        }else{
            for(int i=0; i<size; i++){
                dynamicJobQueue.remove(buf[i]);
            }              
        }   
    }


    void print_proc_nJob(){
        for(int pid=0; pid<worldSize; pid++){
            
            fprintf(stderr, "%d, ", procCtrlTbl[pid].jobQueue.size());
        }
        fprintf(stderr, "\n");
    }


    bool check_send_terminate_signal(){

        // print_proc_nJob();

        for(int pid = 0; pid < worldSize; pid++){
            if(procCtrlTbl[pid].jobQueue.size()) return false;
        }

        int sigBuf = 1;
        for(int pid = 1; pid < worldSize; pid++){
            MPI_Send(&sigBuf, 1, MPI_INT, pid, MPI_TAG_TERMINATE, MPI_COMM_WORLD);
        }
        return true;
    }



    ThreadManager *tmPtr;
    
    int jobDoneBuf[BUF_SIZE_JOB_UPLOAD];
    int jobSchdlBuf[BUF_SIZE_JOB_SCHDL];

    int rank, worldSize;
    int startIdx, stopIdx, nStaticJobs; 
    
    unsigned char *raw_image;
    unsigned char **image;
    char *filename;

    PCB* procCtrlTbl;
    unordered_map<jobIdx_t, JobNode *> jobNodeTbl;
    
    int nJobsCmpltd;
    int* recvCnt;

    MapQueue<jobIdx_t> dynamicJobQueue;

};   






class Process{
    public:
    
    Process(char** argv, int rank, int worldSize){

        tmPtr = new ThreadManager(argv, rank, worldSize);

          
        this->rank = rank;
        this->worldSize = worldSize;

        nStaticJobs =  tmPtr->nTotalJobs / worldSize; 
        startIdx = nStaticJobs * rank;
        if(rank < worldSize - 1){stopIdx = nStaticJobs * (rank + 1) - 1;}
        else if(rank == worldSize - 1){stopIdx = tmPtr->nTotalJobs - 1;}
        
        nStaticJobs = stopIdx - startIdx + 1;
        nJobsUploaded = 0;
        

        for(int i=startIdx; i<=stopIdx; i++){
            dynamicJobQueue.push_back(i);
        }

    }


    void dynamic_job_enqueue(){
        if(dynamicJobQueue.empty()) return;

        static int cnt=startIdx;

        if(tmPtr->jobQueue.unsafe_size() < 10 * tmPtr->nThrds){
            for(int i=0; i < 10 * tmPtr->nThrds; i++){
                
                
         
                jobIdx_t jobIdx;
                if(dynamicJobQueue.try_pop_front(&jobIdx)){
                    Job job;
                    job.idx = jobIdx;

                    assert(jobIdx == cnt);
                    assert(cnt <= stopIdx);
                    cnt++;
                    // fprintf(stderr, "[proc %d] jobIdx: %d, cnt: %d\n", rank, jobIdx, cnt);

                    tmPtr->jobQueue.push(job);
                }
            }
        }
    }



    void start_process(){


        dynamic_job_enqueue();      
  
        tmPtr->start_thread();

        
        while(1){

            if(!tmPtr->jobQueue.empty()){

                tmPtr->one_job();                 

                while(tmPtr->jobDoneQueue.unsafe_size() >= 2500){
                    upload_completed_jobs();
                }
            }
            else{

                upload_completed_jobs();
                check_receive_terminate_signal();
            }

            // dynamic_job_cancel();
            // dynamic_job_receive();
        
            dynamic_job_enqueue();
            // fprintf(stderr, "[proc %d][start_process()] jobQueue size: %d\n",
                                        //  rank, tmPtr->jobQueue.unsafe_size());            


            // fprintf(stderr, "[proc %d][start_process()]jobDoneQueue size: %d\n",
            //                              rank, tmPtr->jobDoneQueue.unsafe_size());  
        }  
    }



    void dynamic_job_receive(){

        int flag, recvSz;
        MPI_Status status;
        
        MPI_Iprobe(0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, &flag, &status);
        if(!flag){return;}        

        MPI_Get_count(&status, MPI_INT, &recvSz);
        if(recvSz > BUF_SIZE_JOB_SCHDL){
            fprintf(stderr, "\n[proc %d][check_job_assignment()] recvSz > jobBufSz.\
                                                         Terminating...\n\n", rank);
            exit(1);            
        }

		MPI_Recv(jobSchdlBuf, recvSz, MPI_INT, 0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        fprintf(stderr, "[proc %d][dynamic_job_receive()] recvSz: %d, %d - %d\n",
                     rank, recvSz, jobSchdlBuf[0], jobSchdlBuf[recvSz-1]);
        

        // fprintf(stderr, "[proc %d][dynamic_job_receive()] jobSchdlBuf[]:\n", rank);                                                                                                                             
        for(int i = 0; i < recvSz; i++){
            // fprintf(stderr, "%d, ", jobSchdlBuf[i]);          
           
            dynamicJobQueue.push_back(jobSchdlBuf[i]);
        }
    }




    void dynamic_job_cancel(){
        int flag, recvSz;
        MPI_Status status;
        
        MPI_Iprobe(0, MPI_TAG_JOB_CANCEL, MPI_COMM_WORLD, &flag, &status);
        if(!flag){return;}        

        MPI_Get_count(&status, MPI_INT, &recvSz);
        if(recvSz > BUF_SIZE_JOB_SCHDL){
            fprintf(stderr, "\n[check_job_assignment()][proc %d] recvSz > jobBufSz.\
                                                         Terminating...\n\n", rank);
            exit(1);            
        }



		MPI_Recv(jobSchdlBuf, recvSz, MPI_INT, 0, MPI_TAG_JOB_CANCEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        fprintf(stderr, "[proc %d][dynamic_job_cancel()] recvSz: %d, %d - %d\n",
                     rank, recvSz, jobSchdlBuf[0], jobSchdlBuf[recvSz-1]);


        for(int i = 0; i < recvSz; i++){
        
            dynamicJobQueue.remove(jobSchdlBuf[i]);
        }
    }



    void upload_completed_jobs(){

        Job job;
        int ofst = 0; 

        while(!tmPtr->jobDoneQueue.empty()){
            if(!tmPtr->jobDoneQueue.try_pop(job)){
                fprintf(stderr, "[pid %d][upload_completed_jobs()] pop failed.\n", rank);
                break;
            }

        //     fprintf(stderr, "[proc %d][upload_completed_jobs()] jobIdx: %d, data: (%f, %f, %f).\n", 
        //   rank, job.idx, job.result.r, job.result.g, job.result.b); 

            jobDoneBuf[ofst++] = job.idx;
            jobDoneBuf[ofst++] = (((int)((unsigned char)job.result.r)) << 24) | \
                                 (((int)((unsigned char)job.result.g)) << 16) | \
                                 (((int)((unsigned char)job.result.b)) <<  8) | \
                                 ((int)255);       

            if(ofst >= BUF_SIZE_JOB_UPLOAD) break;
        }

        if(ofst > 0){
            nJobsUploaded += ofst/2;
            MPI_Send(&jobDoneBuf, ofst, MPI_INT, 0, MPI_TAG_JOB_UPLOAD, MPI_COMM_WORLD);
        } 
    }



    void check_receive_terminate_signal(){

        int flag, recvSz, terminate = 0;
        MPI_Status status;
        
        MPI_Iprobe(0, MPI_TAG_TERMINATE, MPI_COMM_WORLD, &flag, &status);
        if(!flag){return;}        

        MPI_Get_count(&status, MPI_INT, &recvSz);
        if(recvSz != 1){
            fprintf(stderr, "\n[check_receive_terminate_signal()] recvSz != 1.\n\n");
            exit(1);
        }

		MPI_Recv(&terminate, recvSz, MPI_INT, 0, MPI_TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if(terminate){
            fprintf(stderr, "\n[check_receive_terminate_signal()][proc %d] receive\
                                         terminate signal. Terminating...\n\n", rank);
            exit(0);
        }
    }

    




    ThreadManager *tmPtr;

    int jobSchdlBuf[BUF_SIZE_JOB_SCHDL];
    int jobDoneBuf[BUF_SIZE_JOB_UPLOAD];

    MapQueue<jobIdx_t> dynamicJobQueue;

    int rank, worldSize;
    int startIdx, stopIdx, nStaticJobs;  
    int nJobsUploaded;  
    
};    





void write_png(ProcessManager* pm){

    fprintf(stdout, "[write_png()] a\n");

    unsigned error = lodepng_encode32_file(pm->filename, pm->raw_image, pm->tmPtr->width, pm->tmPtr->height);

    fprintf(stdout, "[write_png()] b\n");

    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));

    fprintf(stdout, "[write_png()] c\n");
}



int main(int argc, char** argv) {
    
    assert(argc == 11);

	int rank, worldSize;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    
    // exit(1);

    if(rank == 0){

        ProcessManager pm(argv, rank, worldSize);
        pm.start_process();

        fprintf(stdout, "[proc %d][main()] b\n", rank);

        // pm.write_png();
        write_png(&pm);

        fprintf(stdout, "[proc %d][main()] c\n", rank);
    }
    else{
        Process proc(argv, rank, worldSize);
        proc.start_process();
        
    }

    fprintf(stdout, "[proc %d][main()] d\n", rank);

   

    return 0;
}