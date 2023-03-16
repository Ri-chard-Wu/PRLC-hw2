#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>


#define GLM_FORCE_SWIZZLE  

#define MPI_TAG_JOB_ASSIGN 0
#define MPI_TAG_JOB_CANCEL 1
#define MPI_TAG_TERMINATE 2
#define MPI_TAG_JOB_UPLOAD 3


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

// unsigned int num_threads;  
// unsigned int width;        
// unsigned int height;       
vec2 iResolution;          

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

// vec3 camera_pos; 
// vec3 target_pos; 

unsigned char* raw_image;  
unsigned char** image;     


void write_png(const char* filename) {
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












typedef int taskId_t;
typedef int taskKey_t;
typedef int jobIdx_t;


struct Task{
    int i;
    int j;
    int m;
    int n;
    vec4 result;
};


struct JCB{
    int i;
    int j;     
    vec4 result;
    int n;
}


struct JobNode{
    jobIdx_t jobIdx;
    JobNode* next;
    JobNode* prev;
}


struct PCB{
    int nJobs;
    JobNode* jobQueueFront;
    JobNode* jobQueueBack;
}



class ThreadManager{
    public:

    ThreadManager(char** argv){

        num_threads = atoi(argv[1]);
        camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
        target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
        width = atoi(argv[8]);
        height = atoi(argv[9]);
        nTotalJobs = width * height;
        // total_pixel = width * height;
    
    }


    void start_thread(){

        nTskThrds = num_threads - 1;
        tskThrds = new pthread_t[nTskThrds];

        for(int i = 0; i < nTskThrds; i++){
            argPtr = new entryArg;
            argPtr->objPtr = this;
            argPtr->tid = i;

            ret = pthread_create(&tskThrds[i], NULL, thread_entry, (void *)argPtr);
            if(ret != 0) printf("create thread failed.\n");
        }
    }


    static void * thread_entry(void *arg) {
        entryArg *argPtr = (entryArg *)arg;

        int tid = argPtr->tid;
        ThreadManager *objPtr = (ThreadManager *)argPtr->objPtr;
        objPtr->task(tid);
        
        return NULL;
    }


    void join_thread()
    {
        for (int i = 0; i < nTskThrds; i++){
            pthread_join(tskThrds[i], NULL);
        }  
    }




    void job2task(int idx){

        Task tsk;
        jobIdx2ImgCoord(idx, &(tsk.i), &(tsk.j));

        for (int m = 0; m < AA; ++m) {
            tsk->m = m;
            for (int n = 0; n < AA; ++n) {
                tsk->n = n;
                tskQueue.push(tsk);
            }
        }
    }



    void task(int tid){

        Task tsk;

        while(1){
            if(!tskQueue.empty()){
                if(!tskQueue.try_pop(tsk)){fprintf(stderr, "[task()] pop failed.\n");}

                partial_AA(&tsk);
                tskDoneQueue.push(tsk);
            }
        }
    }


    void one_task(){

        Task tsk;

        if(!tskQueue.empty()){
            if(!tskQueue.try_pop(tsk)){fprintf(stderr, "[task()] pop failed.\n");}

            partial_AA(&tsk);
            tskDoneQueue.push(tsk);
        }
 
    }


    void aggregate_tasks(){
        
        Task tsk;
        JCB jcb;
        taskKey_t tskKey;
        while(!tskDoneQueue.empty()){
            if(!tskDoneQueue.try_pop(tsk)){fprintf(stderr, "[aggregate_tasks()] pop failed.\n");}

            tskKey = tsk2key(tsk);
            if(is_in_jobCtrlTbl(tskKey)){
                jcb = jobCtrlTbl[tskKey];
                jcb.n++;
                jcb.result += tsk.result;
                jcb.result /= (double)(AA * AA);
                jcb.result *= 255.0;  
                if(jcb.n == tmPtr->nTskPerJob){
                    jobCtrlTbl.erase(tskKey);
                    jobDoneQueue.push(jcb);
                }
                else{
                    jobCtrlTbl[tskKey] = jcb;
                }
            }
            else{
                jcb.i = tsk.i;
                jcb.j = tsk.j;
                jcb.n = 1;
                jcb.result = tsk.result;
                jobCtrlTbl[tskKey] = jcb;
            }
        }
    }


    bool is_in_jobCtrlTbl(taskKey_t tskKey){
        return !(jobCtrlTbl->find(tskKey) == jobCtrlTbl->end());
    }


    taskKey_t tsk2key(Task tsk){
        return tsk.i * ((int)width) + tsk.j; 
    }



    void jobIdx2ImgCoord(int idx, int* i, int* j){
        *i = idx / width;
        *j = idx % width;        
    }

    void imgCoord2JobIdx(int i, int j, int* idx){
        *idx = i * ((int)width) + j;      
    }

    void partial_AA(Task* tsk){

        iResolution = vec2(width, height);

        int i = tsk->i; 
        int j = tsk->j; 
        int m = tsk->m; 
        int n = tsk->n;

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
        tsk->result = vec4(col, 1.);
    }


    int get_task_num(){
        return tskQueue.unsafe_size();
    }


    concurrent_queue<Task> tskQueue;
    concurrent_queue<Task> tskDoneQueue;
    concurrent_queue<JCB> jobDoneQueue;


    unsigned int nTskThrds;
    pthread_t *tskThrds;

    unsigned int num_threads;  
    unsigned int width;        
    unsigned int height;  
    vec3 camera_pos; 
    vec3 target_pos; 
    unsigned int nTotalJobs;

    unordered_map<taskKey_t, JCB> *jobCtrlTbl;

    struct entryArg{
        ThreadManager* objPtr;
        int tid;
    };

}




class ProcessManager{ 
    public:
    
    ProcessManager(char** argv, int rank, int worldSize){
        
        tmPtr = new ThreadManager(argv);
    
    
        rank = rank;
        worldSize = worldSize;
        nStaticJobs =  tmPtr->nTotalJobs / world_size; 
        startIdx = nStaticJobs * rank;
        if(rank < world_size - 1){stopIdx = nStaticJobs * (rank + 1) - 1;}
        else if(rank == world_size - 1){stopIdx = tmPtr->nTotalJobs - 1;}


        jobDoneBufSz = 256;
        jobDoneBuf = new int[jobDoneBufSz];
        procCtrlTbl = new PCB[worldSize];


        raw_image = new unsigned char[tmPtr->width * tmPtr->height * 4];
        image = new unsigned char*[tmPtr->height];

        for (int i = 0; i < tmPtr->height; ++i) {
            image[i] = raw_image + i * tmPtr->width * 4;
        }

        static_job_assignment();
        start();
    }



    void static_job_assignment(){
        // no real msg passing.

        int s, e;

        for(int pid=0; pid<world_size; pid++){
            
            s = nStaticJobs * pid;
            if(pid < world_size - 1){e = nStaticJobs * (pid + 1) - 1;}
            else if(pid == world_size - 1){e = tmPtr->nTotalJobs - 1;}
            
            for(int jobIdx = s; jobIdx <= e; jobIdx++){
                enqueue_job(pid, jobIdx);
            }
        }
    }


    void start(){

        tmPtr->start_thread();

        int tskNum, jobNum;
        for(int i = 0; i < (0.9 * nStaticJobs); i++){
            tmPtr->one_task();                       
        }

        while(1){

            receive_completed_jobs();

            tmPtr->aggregate_tasks();
            local_receive_completed_jobs();

            dynamic_job_assignment();

            check_send_terminate_signal();
            tmPtr->one_task();

        }  
        tm->join_thread();  
    }


    void local_receive_completed_jobs(){

        JCB jcb;
        int ofst = 0, jobIdx; 

        while(!tmPtr->jobDoneQueue.empty()){
            if(!tmPtr->jobDoneQueue.try_pop(jcb)){fprintf(stderr, "[upload_completed_jobs()]\
                                                     pop failed.\n");}

            image[jcb.i][4 * jcb.j + 0] = (unsigned char)jcb.result.r;  
            image[jcb.i][4 * jcb.j + 1] = (unsigned char)jcb.result.g;  
            image[jcb.i][4 * jcb.j + 2] = (unsigned char)jcb.result.b;  
            image[jcb.i][4 * jcb.j + 3] = 255;  
         
        }
    }


    void receive_completed_jobs(){

        int flag, recvSz, i, j, data, jobIdx;
        int dataMask = (1 << 8) - 1;
        PCB* pcbPtr;
        MPI_Status status;
        
        for(int pid=1; pid<world_size; pid++){
            MPI_Iprobe(pid, MPI_TAG_JOB_UPLOAD, MPI_COMM_WORLD, &flag, &status);
            if(!flag){return;}        

            MPI_Get_count(&status, MPI_INT, &recvSz);
            MPI_Recv(jobDoneBuf, *recvSz, MPI_INT, pid, MPI_TAG_JOB_UPLOAD,
                                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int k = 0; k < recvSz; k += 2){
                jobIdx = jobDoneBuf[k];
                tmPtr->jobIdx2ImgCoord(jobIdx, &i, &j);
                data = jobDoneBuf[k+1];

                image[i][4 * j + 0] = (unsigned char)((data >> 24) |  dataMask);  
                image[i][4 * j + 1] = (unsigned char)((data >> 16) |  dataMask);  
                image[i][4 * j + 2] = (unsigned char)((data >> 8 ) |  dataMask);  
                image[i][4 * j + 3] = 255;  

                unqueue_job(pid, jobIdx);

            }
        }
    }


    void enqueue_job(int pid, jobIdx_t jobIdx){
        // - Three things to update:
        //      1. `jobNodeTbl`
        //      2. job queue
        //      3. `procCtrlTbl[pid].nJobs`

        JobNode *backPtr = procCtrlTbl[pid].jobQueueBack;
        
        JobNode *curPtr  = new JobNode;
        curPtr->jobIdx = jobIdx;
        curPtr->next = NULL;
        curPtr->prev = backPtr;
        
        if(backPtr){
            backPtr->next = curPtr;
            procCtrlTbl[pid].jobQueueBack = curPtr;            
        }
        else{ 
            procCtrlTbl[pid].jobQueueFront = curPtr;
            procCtrlTbl[pid].jobQueueBack = curPtr;
        }

        jobNodeTbl[jobIdx] = curPtr;
        procCtrlTbl[pid].nJobs++;
    }



    void unqueue_job(int pid, jobIdx_t jobIdx){
        // - Three things to update:
        //      1. `jobNodeTbl`
        //      2. job queue
        //      3. `procCtrlTbl[pid].nJobs`

        JobNode *curPtr = jobNodeTbl[jobIdx];
        JobNode *fNdPtr, *bNdPtr;

        fNdPtr = curPtr->next;
        bNdPtr = curPtr->prev;

        if(fNdPtr && bNdPtr){
            fNdPtr->prev = bNdPtr;
            bNdPtr->next = fNdPtr;            
        } 
        else if(fNdPtr && !bNdPtr){
            fNdPtr->next = NULL;
            procCtrlTbl[pid].jobQueueBack = fNdPtr;
        }
        else if(!fNdPtr && bNdPtr){
            procCtrlTbl[pid].jobQueueFront = bNdPtr;
            bNdPtr->prev = NULL;
        }
        else{
            procCtrlTbl[pid].jobQueueFront->next = NULL;
            procCtrlTbl[pid].jobQueueFront->next = NULL;
        }

        delete curPtr;
        jobNodeTbl[jobIdx] = NULL;
        procCtrlTbl[pid].nJobs--;
    }



    void dynamic_job_assignment(){
        PCB* pcbPtr;
        int minNJob = 1 << 16, minNJobPid;
        int maxNJob = 0, maxNJobPid;

        for(int pid=0; pid<world_size; pid++){


            if(minNJob > procCtrlTbl[pid].nJobs){
                minNJob = procCtrlTbl[pid].nJobs;
                minNJobPid = pid;
            }

            if(maxNJob < procCtrlTbl[pid].nJobs){
                maxNJob = procCtrlTbl[pid].nJobs;
                maxNJobPid = pid;
            }
        }

        jobIdx_t jobIdx;
        if(maxNJob - minNJob >= 2){
            
            jobIdx = procCtrlTbl[maxNJobPid].jobQueueBack->jobIdx;
            unqueue_job(maxNJobPid, jobIdx);
            enqueue_job(minNJobPid, jobIdx);

            dynamic_cancel_job(maxNJobPid, jobIdx);
            dynamic_assign_job(minNJobPid, jobIdx);
            
        }
    }


    void dynamic_assign_job(int pid, jobIdx_t jobIdx){
        MPI_Send(&jobIdx, 1, MPI_INT, pid, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD);        
    }


    void dynamic_cancel_job(int pid, jobIdx_t jobIdx){
        MPI_Send(&jobIdx, 1, MPI_INT, pid, MPI_TAG_JOB_CANCEL, MPI_COMM_WORLD);        
    }


    void check_send_terminate_signal(){
        for(int pid = 1; pid < world_size; pid++){
            if(procCtrlTbl[pid].nJobs) return;
        }

        int sigBuf = 1;
        for(int pid = 1; pid < world_size; pid++){
            MPI_Send(&sigBuf, 1, MPI_INT, pid, MPI_TAG_TERMINATE, MPI_COMM_WORLD);
        }
    }



    int *jobDoneBuf;
    int jobDoneBufSz; 

    int rank, worldSize;
    int startIdx, stopIdx, nStaticJobs; 
    
    unsigned char *raw_image;
    unsigned char **image;

    PCB* procCtrlTbl;
    unordered_map<jobIdx_t, JobNode *> jobNodeTbl;

}    





class Process{ 
    public:
    
    Process(char** argv, int rank, int worldSize){

        tmPtr = new ThreadManager(argv);

        jobBufSz = 256;
        jobBuf = new int[jobBufSz];

        jobDoneBufSz = 256;
        jobDoneBuf = new int[jobDoneBufSz];
          
        rank = rank;
        worldSize = worldSize;
        nStaticJobs =  tmPtr->nTotalJobs / world_size; 
        startIdx = nStaticJobs * rank;
        if(rank < world_size - 1){stopIdx = nStaticJobs * (rank + 1) - 1;}
        else if(rank == world_size - 1){stopIdx = tmPtr->nTotalJobs - 1;}

        static_job2task();
        start();
    }



    void start(){

        tmPtr->start_thread();

        int tskNum;
        for(int i = 0; i < (0.9 * nStaticJobs); i++){
            tmPtr->one_task();                       
        }

        while(1){

            tmPtr->aggregate_tasks();
            upload_completed_jobs();

            check_enqueue_job_assignment(&jobNum);
            check_receive_terminate_signal();

            tmPtr->one_task();
        }  
        tm->join_thread();  
    }




    void static_job2task(){

        for(int idx = startIdx; idx <= stopIdx; idx++){
            job2task(idx);
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

    

    void check_enqueue_job_assignment(){
        int flag, recvSz;
        MPI_Status status;
        
        MPI_Iprobe(0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, &flag, &status);
        if(!flag){return;}        

        MPI_Get_count(&status, MPI_INT, &recvSz);
        if(recvSz > jobBufSz){
            fprintf(stderr, "\n[check_job_assignment()][proc %d] recvSz > jobBufSz.\
                                                         Terminating...\n\n", rank);
            exit(1);            
        }

		MPI_Recv(jobBuf, recvSz, MPI_INT, 0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i = 0; i < recvSz; i++){
            tmPtr->job2task(jobBuf[i]);
        }
    }


    void upload_completed_jobs(){

        JCB jcb;
        int ofst = 0, jobIdx; 

        while(!tmPtr->jobDoneQueue.empty()){
            if(!tmPtr->jobDoneQueue.try_pop(jcb)){fprintf(stderr, "[upload_completed_jobs()] pop failed.\n");}

            tmPtr->imgCoord2JobIdx(jcb.i, jcb.j, &jobIdx);

            jobDoneBuf[ofst++] = jobIdx;
            jobDoneBuf[ofst++] = (((int)jcb.result.r) << 24) | (((int)jcb.result.g) << 16) | 
                (((int)jcb.result.b) << 8) | ((int)255);       

            if(ofst >= jobDoneBufSz)break;
        }

        if(ofst > 0) MPI_Send(&jobDoneBuf, ofst, MPI_INT, 0, MPI_TAG_JOB_UPLOAD, MPI_COMM_WORLD);
    }




    ThreadManager *tmPtr;
    int *jobBuf;
    int jobBufSz;    
    int *jobDoneBuf;
    int jobDoneBufSz;   

    int rank, worldSize;
    int startIdx, stopIdx, nStaticJobs;    
    
}    












int main(int argc, char** argv) {
    
    assert(argc == 11);

	int rank, world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(rank == 0){
        ProcessManager pm(argv, rank, world_size);
    }
    else{
        Process proc(argv, rank, world_size);
    }


    write_png(argv[10]);

    delete[] raw_image;
    delete[] image;
   

    return 0;
}
