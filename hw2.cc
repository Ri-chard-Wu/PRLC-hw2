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
#define BUF_SIZE_JOB_UPLOAD 5000
// #define BUF_SIZE_JOB_ASSIGN 1000

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
};


struct JobNode{
    jobIdx_t jobIdx;
    JobNode* next;
    JobNode* prev;
};


struct PCB{
    int nJobs;
    JobNode* jobQueueFront;
    JobNode* jobQueueBack;
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
        nTskPerJob = AA * AA;  


        
    }


    void start_thread(){

        nTskThrds = num_threads - 1;
        tskThrds = new pthread_t[nTskThrds];
        entryArg* argPtr;
        int ret;

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

        fprintf(stdout, "[proc %d][join_thread()] all threads joined.\n", rank);
    }




    void job2task(int idx){

        Task tsk;
        jobIdx2ImgCoord(idx, &(tsk.i), &(tsk.j));

        for (int m = 0; m < AA; ++m) {
            tsk.m = m;
            for (int n = 0; n < AA; ++n) {
                tsk.n = n;
                tskQueue.push(tsk);
            }
        }
    }



    void task(int tid){

        Task tsk;


        while(!tskQueue.empty()){
            
            if(!tskQueue.try_pop(tsk)){fprintf(stderr, "[task()] pop failed.\n");}

            partial_AA(&tsk);
            tskDoneQueue.push(tsk);
           
        }


        // while(1){
        //     if(!tskQueue.empty()){
        //         if(!tskQueue.try_pop(tsk)){fprintf(stderr, "[task()] pop failed.\n");}

        //         partial_AA(&tsk);
        //         tskDoneQueue.push(tsk);
        //     }
        // }
    }


    void one_task(){

        Task tsk;
        
        // fprintf(stderr, "[proc %d][one_task()] c\n", rank);

        if(!tskQueue.empty()){
            if(!tskQueue.try_pop(tsk)){fprintf(stderr, "[task()] pop failed.\n");}

            // if(rank==0)fprintf(stderr, "[proc %d][one_task()] d\n", rank);

            partial_AA(&tsk);
            tskDoneQueue.push(tsk);
        }

        // fprintf(stderr, "[proc %d][one_task()] e\n", rank);
    }


    void aggregate_tasks(){
        
        Task tsk;
        JCB jcb;
        jobIdx_t jobIdx;

        while(!tskDoneQueue.empty()){
            if(!tskDoneQueue.try_pop(tsk)){fprintf(stderr, "[aggregate_tasks()] pop failed.\n");}

            imgCoord2JobIdx(tsk.i, tsk.j, &jobIdx);

            if(is_in_jobCtrlTbl(jobIdx)){

                jcb = jobCtrlTbl[jobIdx];
                jcb.n++;
                jcb.result += tsk.result;
                if(jcb.n == nTskPerJob){
                    jcb.result /= (double)(nTskPerJob);
                    jcb.result *= 255.0;                     
                    jobDoneQueue.push(jcb);
                    // jobCtrlTbl.erase(jobIdx);
                }
                else if(jcb.n < nTskPerJob){
                    jobCtrlTbl[jobIdx] = jcb;
                }
                else{
                    fprintf(stderr, "[aggregate_tasks()] jcb.n > nTskPerJob." );
                    exit(1);
                }

            }
            else{
                jcb.i = tsk.i;
                jcb.j = tsk.j;
                jcb.n = 1;
                jcb.result = tsk.result;
                jobCtrlTbl[jobIdx] = jcb;
            }
        }
    }


    bool is_in_jobCtrlTbl(jobIdx_t jobIdx){
        return !(jobCtrlTbl.find(jobIdx) == jobCtrlTbl.end());
    }



    void jobIdx2ImgCoord(int idx, int* i, int* j){
        *i = idx / width;
        *j = idx % width;        
    }

    void imgCoord2JobIdx(int i, int j, int* idx){
        *idx = i * ((int)width) + j;      
    }

    void partial_AA(Task* tsk){

        vec2 iResolution = vec2(width, height);

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





    int rank, worldSize;

    concurrent_queue<Task> tskQueue;
    concurrent_queue<Task> tskDoneQueue;
    queue<JCB> jobDoneQueue;


    unsigned int nTskThrds;
    pthread_t *tskThrds;
    

    unsigned int num_threads;  
    unsigned int width;        
    unsigned int height;  
    vec3 camera_pos; 
    vec3 target_pos; 
    unsigned int nTotalJobs;
    int nTskPerJob;

    unordered_map<jobIdx_t, JCB> jobCtrlTbl;

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
        for(int i=0;i<worldSize;i++){
            procCtrlTbl[i].jobQueueFront = NULL;
            procCtrlTbl[i].jobQueueBack = NULL;
            procCtrlTbl[i].nJobs = 0;
        }


        raw_image = new unsigned char[tmPtr->width * tmPtr->height * 4];
        image = new unsigned char*[tmPtr->height];

        for (int i = 0; i < tmPtr->height; ++i) {
            image[i] = raw_image + i * tmPtr->width * 4;
        }

        static_job2task();
        init_job_enqueue();
        
    }



    void init_job_enqueue(){
        // no real msg passing.
        int s, e;

        for(int pid=0; pid < worldSize; pid++){
            
            s = nStaticJobs * pid;
            if(pid < worldSize - 1){e = nStaticJobs * (pid + 1) - 1;}
            else if(pid == worldSize - 1){e = tmPtr->nTotalJobs - 1;}
            else{
                fprintf(stderr, "[proc 0][init_job_enqueue()] Invalid pid: %d\n", pid);
                exit(1);                
            }
            
            for(int jobIdx = s; jobIdx <= e; jobIdx++){
                enqueue_job(pid, jobIdx);
            }

        }   
        // fprintf(stderr, "[proc 0][init_job_enqueue()] tmPtr->nTotalJobs: %d\n", tmPtr->nTotalJobs);

        // print_proc_nJob();
    }

    void static_job2task(){
        for(int idx = startIdx; idx <= stopIdx; idx++){
            tmPtr->job2task(idx);
        }    
    }


    void start_process(){
        // print_proc_nJob();
        tmPtr->start_thread();

        int nStaticTsks = nStaticJobs * tmPtr->nTskPerJob;

        // while(tmPtr->tskQueue.unsafe_size() > 0.1 * nStaticTsks){
        while(!tmPtr->tskQueue.empty()){
     
            for(int i=0;i<1;i++){
                tmPtr->one_task();                 
            }
            
            // while(tmPtr->tskDoneQueue.unsafe_size() >= 100000){

            //     tmPtr->aggregate_tasks();

            //     local_receive_completed_jobs();
                
            // }

            receive_completed_jobs();
        }  


        tmPtr->join_thread(); 
        while(nJobsCmpltd < nStaticJobs){
            tmPtr->aggregate_tasks();
            local_receive_completed_jobs();
        }
        

        print_queue_sizes();


        while(!check_send_terminate_signal()){
            receive_completed_jobs();
        }  

        fprintf(stderr, "[proc 0][start] a\n");
        MPI_Finalize();
        fprintf(stderr, "[proc 0][start] b\n");


    }

    void print_queue_sizes(){
        fprintf(stderr, "[proc %d] queue sizes:   %d, %d, %d\n", rank, 
        tmPtr->tskQueue.unsafe_size(), tmPtr->tskDoneQueue.unsafe_size(), 
        tmPtr->jobDoneQueue.size());
    }


    void print_proc_nJob(){
        
        for(int pid=0; pid<worldSize; pid++){
            
            fprintf(stdout, "%d, ", procCtrlTbl[pid].nJobs);
        }

        fprintf(stdout, "\n");

    }




    void local_receive_completed_jobs(){

        JCB jcb;
        int ofst = 0, jobIdx; 

        while(!tmPtr->jobDoneQueue.empty()){
            // if(!tmPtr->jobDoneQueue.try_pop(jcb)){fprintf(stderr, "[upload_completed_jobs()]\
            //                                          pop failed.\n");}

            jcb = tmPtr->jobDoneQueue.front();
            tmPtr->jobDoneQueue.pop();


            tmPtr->imgCoord2JobIdx(jcb.i, jcb.j, &jobIdx);

            image[jcb.i][4 * jcb.j + 0] = (unsigned char)jcb.result.r;  
            image[jcb.i][4 * jcb.j + 1] = (unsigned char)jcb.result.g;  
            image[jcb.i][4 * jcb.j + 2] = (unsigned char)jcb.result.b;  
            image[jcb.i][4 * jcb.j + 3] = (unsigned char)255;  

            unqueue_job(0, jobIdx);
            ofst++;
         
        }
        
        if(ofst > 0){
            nJobsCmpltd += ofst;
            // fprintf(stdout, "[proc %d][local_receive_completed_jobs()] from pid 0 recvSz: %d\n", rank, ofst);  
            fprintf(stdout, "[proc %d][receive_completed_jobs()] from pid 0 recvSz: %d\n",
            rank, nJobsCmpltd);   

            fprintf(stderr, "[proc %d][receive_completed_jobs()] from pid 0 recvSz: %d\n",
            rank, nJobsCmpltd);                                   
        }
        


    }

    int* recvCnt;
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

            fprintf(stderr, "[proc %d][receive_completed_jobs()] from pid %d recvSz: %d\n",
            rank, pid, recvCnt[pid]/2); 

            for(int k = 0; k < recvSz; k += 2){
                jobIdx = jobDoneBuf[k];
                tmPtr->jobIdx2ImgCoord(jobIdx, &i, &j);
                data = jobDoneBuf[k+1];

                image[i][4 * j + 0] = (unsigned char)((data >> 24) &  dataMask);  
                image[i][4 * j + 1] = (unsigned char)((data >> 16) &  dataMask);  
                image[i][4 * j + 2] = (unsigned char)((data >> 8 ) &  dataMask);  
                image[i][4 * j + 3] = (unsigned char)255;  

                if(jobIdx < 0 || jobIdx >= (tmPtr->nTotalJobs)){
                    fprintf(stderr, "[proc %d][receive_completed_jobs()] job idx out of range.\n", rank);            
                    exit(1);
                }

                unqueue_job(pid, jobIdx);

            }
        }
    }


    void enqueue_job(int pid, jobIdx_t jobIdx){
        // - Three things to update:
        //      1. `jobNodeTbl`
        //      2. job queue
        //      3. `procCtrlTbl[pid].nJobs`

        // fprintf(stderr, "[proc 0][enqueue_job()] pid: %d, jobIdx: %d\n", pid, jobIdx);

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

        if(!curPtr){
            fprintf(stderr, "[proc %d][unqueue_job()] !curPtr. pid: %d, jobIdx: %d\n", rank, pid, jobIdx);
            // exit(1);
            return;
        }

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
            // fprintf(stderr, "[proc 0][unqueue_job()] back jobIdx: %d\n",
            //      rank,  procCtrlTbl[pid].jobQueueBack->jobIdx);
        }
        else if(!fNdPtr && bNdPtr){
            procCtrlTbl[pid].jobQueueFront = bNdPtr;
            bNdPtr->prev = NULL;
        }
        else{
            procCtrlTbl[pid].jobQueueFront = NULL;
            procCtrlTbl[pid].jobQueueBack = NULL;
        }

        delete curPtr;
        jobNodeTbl[jobIdx] = NULL;
        procCtrlTbl[pid].nJobs--;
    }



    void dynamic_job_assignment(){

        PCB* pcbPtr;
        int minNJob = 1 << 16, minNJobPid;
        int maxNJob = 0, maxNJobPid;

        for(int pid=0; pid<worldSize; pid++){


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

            fprintf(stdout, "[dynamic_job_assignment()] maxNJob: %d, maxNJobPid: %d, minNJob: %d, minNJobPid: %d\n",
                             maxNJob, maxNJobPid, minNJob, minNJobPid);

            // fprintf(stderr, "[proc %d][dynamic_job_assignment()] a\n", rank);

            unqueue_job(maxNJobPid, jobIdx);

            // fprintf(stderr, "[proc %d][dynamic_job_assignment()] b\n", rank);

            enqueue_job(minNJobPid, jobIdx);

            // fprintf(stderr, "[proc %d][dynamic_job_assignment()] c\n", rank);

            // dynamic_cancel_job(maxNJobPid, jobIdx);
            dynamic_assign_job(minNJobPid, jobIdx);
        }
    }


    void dynamic_assign_job(int pid, jobIdx_t jobIdx){
        if(pid != 0){
            MPI_Send(&jobIdx, 1, MPI_INT, pid, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD);
        }else{
            tmPtr->job2task(jobIdx);
        }            
                
    }


    void dynamic_cancel_job(int pid, jobIdx_t jobIdx){
               
        if(pid != 0){
            MPI_Send(&jobIdx, 1, MPI_INT, pid, MPI_TAG_JOB_CANCEL, MPI_COMM_WORLD); 
        }else{
            
        }   
    }


    bool check_send_terminate_signal(){

        // print_proc_nJob();

        for(int pid = 1; pid < worldSize; pid++){
            if(procCtrlTbl[pid].nJobs) return false;
        }

        int sigBuf = 1;
        for(int pid = 1; pid < worldSize; pid++){
            MPI_Send(&sigBuf, 1, MPI_INT, pid, MPI_TAG_TERMINATE, MPI_COMM_WORLD);
        }
        return true;
    }


    // void write_png(){

    //     fprintf(stdout, "[proc %d][write_png()] a\n", rank);

    //     fprintf(stdout, "[proc %d][write_png()] width: %d, height: %d\n", rank, tmPtr->width, tmPtr->height);

    //     unsigned error = lodepng_encode32_file(filename, raw_image, tmPtr->width, tmPtr->height);

    //     fprintf(stdout, "[proc %d][write_png()] b\n", rank);

    //     if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));

    //     fprintf(stdout, "[proc %d][write_png()] c\n", rank);
    // }


    ThreadManager *tmPtr;
    
    int jobDoneBuf[BUF_SIZE_JOB_UPLOAD];
    int jobDoneBufSz = BUF_SIZE_JOB_UPLOAD;

    int rank, worldSize;
    int startIdx, stopIdx, nStaticJobs; 
    
    unsigned char *raw_image;
    unsigned char **image;
    char *filename;

    PCB* procCtrlTbl;
    unordered_map<jobIdx_t, JobNode *> jobNodeTbl;
    int nJobsCmpltd;

};   





class Process{
    public:
    
    Process(char** argv, int rank, int worldSize){

        // fprintf(stderr, "[proc %d][Process()] \n", rank);
        // exit(1);

        tmPtr = new ThreadManager(argv, rank, worldSize);

          
        this->rank = rank;
        this->worldSize = worldSize;
        nStaticJobs =  tmPtr->nTotalJobs / worldSize; 
        startIdx = nStaticJobs * rank;
        if(rank < worldSize - 1){stopIdx = nStaticJobs * (rank + 1) - 1;}
        else if(rank == worldSize - 1){stopIdx = tmPtr->nTotalJobs - 1;}
        nStaticJobs = stopIdx - startIdx + 1;
        nJobsUploaded = 0;

        // fprintf(stderr, "[proc %d][Process()] \n", rank);

        static_job2task();
    }


    void print_queue_sizes(){
        fprintf(stderr, "[proc %d] queue sizes:   %d, %d, %d\n", rank, 
        tmPtr->tskQueue.unsafe_size(), tmPtr->tskDoneQueue.unsafe_size(), 
        tmPtr->jobDoneQueue.size());
    }


    void start_process(){

        tmPtr->start_thread();



        int nStaticTsks = nStaticJobs * tmPtr->nTskPerJob;
        
        while(!tmPtr->tskQueue.empty()){
     
            for(int i=0;i<1;i++){
                tmPtr->one_task();                 
            }
            
            // while(tmPtr->tskDoneQueue.unsafe_size() >= 0.01 * nStaticTsks){
            while(tmPtr->tskDoneQueue.unsafe_size() >= 10000){
                tmPtr->aggregate_tasks();
                // while(tmPtr->jobDoneQueue.size() >= 0.01 * nStaticJobs){
                // while(tmPtr->jobDoneQueue.size() >= 2500){
                upload_completed_jobs();
                
                // }                  
            }

            // print_queue_sizes();
            
        }  


        tmPtr->join_thread();    
        while(nJobsUploaded < nStaticJobs){
            tmPtr->aggregate_tasks();
            upload_completed_jobs();
        }
  



        fprintf(stderr, "[proc %d][start] a\n", rank);
        MPI_Finalize();
        fprintf(stderr, "[proc %d][start] b\n", rank);

        // check_enqueue_job_assignment();
        // check_receive_terminate_signal();
        // fprintf(stdout, "[proc %d][start()] f\n", rank);
    }






    void static_job2task(){
        
        // fprintf(stderr, "[proc %d][static_job2task()] \n", rank);

        for(int idx = startIdx; idx <= stopIdx; idx++){
            tmPtr->job2task(idx);
        }    

        // fprintf(stderr, "[proc %d][static_job2task()] \n", rank);
        // exit(1);        
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

    

    // void check_enqueue_job_assignment(){
    //     int flag, recvSz;
    //     MPI_Status status;
        
    //     MPI_Iprobe(0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, &flag, &status);
    //     if(!flag){return;}        

    //     MPI_Get_count(&status, MPI_INT, &recvSz);
    //     if(recvSz > jobBufSz){
    //         fprintf(stderr, "\n[check_job_assignment()][proc %d] recvSz > jobBufSz.\
    //                                                      Terminating...\n\n", rank);
    //         exit(1);            
    //     }

	// 	MPI_Recv(jobBuf, recvSz, MPI_INT, 0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    //     fprintf(stderr, "\n[proc %d][check_job_assignment()] recvSz: %d", rank, recvSz);
                                                                
    //     for(int i = 0; i < recvSz; i++){
    //         tmPtr->job2task(jobBuf[i]);
    //         fprintf(stderr, "\n[proc %d][check_job_assignment()] jobIdx: %d", rank, jobBuf[i]);

    //     }
    // }


    void upload_completed_jobs(){

        JCB jcb;
        int ofst = 0, jobIdx; 
        // int acc_ofst = 0;

        while(!tmPtr->jobDoneQueue.empty()){
            // if(!tmPtr->jobDoneQueue.try_pop(jcb)){fprintf(stderr, "[upload_completed_jobs()] pop failed.\n");}
            jcb = tmPtr->jobDoneQueue.front();
            tmPtr->jobDoneQueue.pop();

            tmPtr->imgCoord2JobIdx(jcb.i, jcb.j, &jobIdx);

            // fprintf(stderr, "[proc %d][upload_completed_jobs()]\
            //                      i: %d, j: %d, jobIdx: %d\n", rank, jcb.i, jcb.j, jobIdx);

            jobDoneBuf[ofst++] = jobIdx;

            // fprintf(stdout, "[proc %d][upload_completed_jobs()] jcb.result.r: %d\n", rank, (int)jcb.result.r);

            jobDoneBuf[ofst++] = (((int)jcb.result.r) << 24) | (((int)jcb.result.g) << 16) | 
                (((int)jcb.result.b) << 8) | ((int)255);       

            if(ofst >= jobDoneBufSz)break;
        }

        if(ofst > 0){
            nJobsUploaded += ofst/2;
            // fprintf(stdout, "[proc %d][upload_completed_jobs()] ofst: %d\n", rank, ofst);
            

            MPI_Send(&jobDoneBuf, ofst, MPI_INT, 0, MPI_TAG_JOB_UPLOAD, MPI_COMM_WORLD);
        } 


        // if(acc_ofst == 2 * nStaticJobs){
        //     fprintf(stdout, "[proc %d][upload_completed_jobs()] all %d static jobs has been uploaded", acc_ofst);
        // }

    }




    ThreadManager *tmPtr;
    // int jobBuf[BUF_SIZE_JOB_ASSIGN];
    // int jobBufSz = BUF_SIZE_JOB_ASSIGN;    

    int jobDoneBuf[BUF_SIZE_JOB_UPLOAD];
    int jobDoneBufSz = BUF_SIZE_JOB_UPLOAD;   

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

   

    return 0;
}
