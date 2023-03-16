#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>


#define GLM_FORCE_SWIZZLE  

#define MPI_TAG_JOB_ASSIGN 0
#define MPI_TAG_TERMINATE 1
#define MPI_TAG_JOB_UPLOAD 2


#include <glm/glm.hpp>


#include <mpi.h>
#include<pthread.h>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_queue.h"
#include <unordered_map>
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



class ThreadManager{
    public:

    ThreadManager(char** argv){

        num_threads = atoi(argv[1]);
        camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
        target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
        width = atoi(argv[8]);
        height = atoi(argv[9]);
        nTotalTsk = width * height;
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
       
        tsk->i = idx / width;
        tsk->j = idx % width;
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

    jobIdx_t jcb2Idx(JCB jcb){
        return jcb.i * ((int)width) + jcb.j; 
    }

    void partial_AA(Task* tsk){

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
    unsigned int nTotalTsk;

    unordered_map<taskKey_t, JCB> *jobCtrlTbl;

    struct entryArg{
        ThreadManager* objPtr;
        int tid;
    };

}




class ProcessManager{ 
    public:
    
    ProcessManager(char** argv, int rank, int worldSize){
        
        jobDoneBufSz = 256;
        jobDoneBuf = new int[jobDoneBufSz];
        worldSize = worldSize;
          
    }


    void receive_completed_jobs(){

        int flag, recvSz;
        MPI_Status status;
        
        for(int i=1; i<world_size; i++){
            MPI_Iprobe(i, MPI_TAG_JOB_UPLOAD, MPI_COMM_WORLD, &flag, &status);
            if(!flag){return;}        

            MPI_Get_count(&status, MPI_INT, &recvSz);
            MPI_Recv(jobDoneBuf, *recvSz, MPI_INT, i, MPI_TAG_JOB_UPLOAD,
                                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
  
    }


    void dynamic_job_assignment(){

    }


    void send_terminate_signal(){

    }



    int *jobDoneBuf;
    int jobDoneBufSz; 

    int worldSize;

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
        nStaticJobs =  tmPtr->nTotalTsk / world_size; 
        startIdx = nStaticJobs * rank;
        if(rank < world_size - 1){stopIdx = nStaticJobs * (rank + 1) - 1;}
        else if(rank == world_size - 1){stopIdx = tmPtr->nTotalTsk - 1;}

        static_job2task();
        tmPtr->start_thread();
        start();
    }



    void start(){

        int tskNum, jobNum;

        for(int i = 0; i < (0.9 * nStaticJobs); i++){
            tmPtr->one_task();                       
        }

        while(1){

            tmPtr->aggregate_tasks();
            upload_completed_jobs();

            check_job_assignment(&jobNum);
            tskNum = tmPtr->get_task_num();

            if(tskNum == 0){
                while(jobNum == 0){
                    check_termination();
                    check_job_assignment(&jobNum);
                }
            }
            tmPtr->one_task();
        }  
        tm->join_thread();  
    }




    void static_job2task(){

        for(int idx = startIdx; idx <= stopIdx; idx++){
            job2task(idx);
        }            
    }



    void check_termination(){

        int flag, recvSz, terminate = 0;
        MPI_Status status;
        
        MPI_Iprobe(0, MPI_TAG_TERMINATE, MPI_COMM_WORLD, &flag, &status);
        if(!flag){return;}        

        MPI_Get_count(&status, MPI_INT, &recvSz);
        if(recvSz != 1){
            fprintf(stderr, "\n[check_termination()] recvSz != 1.\n\n");
            exit(1);
        }

		MPI_Recv(&terminate, recvSz, MPI_INT, 0, MPI_TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if(terminate){
            fprintf(stderr, "\n[check_termination()][proc %d] receive\
                                         terminate signal. Terminating...\n\n", rank);
            exit(0);
        }
    }

    

    void check_job_assignment(int* recvSz){
        int flag;
        MPI_Status status;
        
        MPI_Iprobe(0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, &flag, &status);
        if(!flag){return;}        

        MPI_Get_count(&status, MPI_INT, recvSz);
        if(recvSz > jobBufSz){
            fprintf(stderr, "\n[check_job_assignment()][proc %d] recvSz > jobBufSz.\
                                                         Terminating...\n\n", rank);
            exit(1);            
        }

		MPI_Recv(jobBuf, *recvSz, MPI_INT, 0, MPI_TAG_JOB_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i = 0; i < *recvSz; i++){
            tmPtr->job2task(jobBuf[i]);
        }
    }


    void upload_completed_jobs(){

        JCB jcb;
        int ofst = 0, tskKey; 

        while(!tmPtr->jobDoneQueue.empty()){
            if(!tmPtr->jobDoneQueue.try_pop(jcb)){fprintf(stderr, "[upload_completed_jobs()] pop failed.\n");}

            tskKey = jcb2Idx(jcb);
            jobDoneBuf[ofst++] = tskKey;
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
        
        // ProcessManager pm(argv, rank, world_size);
        // Process proc(argv, rank, world_size);

    }
    else{
        Process proc(argv, rank, world_size);
    }








    //###############################################




    ThreadManager tm(argv, rank, world_size);
    if(rank == 0){
        ProcessManager pm(argv, rank, world_size);
    }


    tm.job2task();
    tm.start_thread();
    
    bool hasTask;
    while(1){

        hasTask = tm.one_task();
        if(!hasTask) return;
        tm.aggregate_tasks();

        if(rank == 0){
            pm.dynamic_scheduling();
            pm.aggregate_jobs();
        }
                      
    }
    
    if(rank == 0){
        pm.aggregate_jobs();
    }


    tm.join_thread(); 




    // num_threads = atoi(argv[1]);
    // camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
    // target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
    // width = atoi(argv[8]);
    // height = atoi(argv[9]);


    // double total_pixel = width * height;
    // double current_pixel = 0;




    iResolution = vec2(width, height);

    raw_image = new unsigned char[width * height * 4];
    image = new unsigned char*[height];

    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * 4;
    }
 



    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            vec4 fcol(0.), partial_fcol(0.);  

            for (int m = 0; m < AA; ++m) {
                for (int n = 0; n < AA; ++n) {

                    partial_AA(i, j, m, n, &partial_fcol);

                    fcol += partial_fcol;
                }
            }

            fcol /= (double)(AA * AA);
            fcol *= 255.0;

            image[i][4 * j + 0] = (unsigned char)fcol.r;  
            image[i][4 * j + 1] = (unsigned char)fcol.g;  
            image[i][4 * j + 2] = (unsigned char)fcol.b;  
            image[i][4 * j + 3] = 255;                    

            // current_pixel++;
            // printf("rendering...%5.2lf%%\r", current_pixel / total_pixel * 100.);
        }
    }




    write_png(argv[10]);

    delete[] raw_image;
    delete[] image;
   

    return 0;
}
