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
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>


#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning SSE2 support is not available. Code will not compile
#endif

using namespace std::chrono;
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


void md2(vec3 p0, vec3 p1, __m128d& trap2, __m128d& out2) {

    __m128d bailout2 = _mm_set_pd(bailout, bailout);
    double r0_out = 0, r1_out = 0;

    // vec3 v = p;
    __m128d vx2 = _mm_set_pd(p0.x, p1.x);
    __m128d vy2 = _mm_set_pd(p0.y, p1.y);
    __m128d vz2 = _mm_set_pd(p0.z, p1.z);

    // double dr = 1.;
    __m128d dr2 = _mm_set1_pd(1.)

    double buf_pd[2];         

    // double r = glm::length(v);
    // trap = r;
    __m128d vx2_sq = _mm_mul_pd(vx2, vx2);
    __m128d vy2_sq = _mm_mul_pd(vy2, vy2);
    __m128d vz2_sq = _mm_mul_pd(vz2, vz2);
    __m128d r2 = _mm_sqrt_pd(_mm_add_pd(_mm_add_pd(vx2_sq, vy2_sq), vz2_sq));
    trap2 = r2;


    for (int i = 0; i < md_iter; ++i) {

        // double theta = glm::atan(v.y, v.x) * power;
        __m128d vy_over_vx = _mm_div_pd(vy2, vx2);
        _mm_storeu_pd(buf_pd, vy_over_vx);    
        __m128d atrig2 = _mm_set_pd(glm::atan(buf_pd[1]), glm::atan(buf_pd[0]));
        __m128d power2 = _mm_set_pd(power, power);
        __m128d theta2 = _mm_mul_pd(atrig2, power2);

        // double phi = glm::asin(v.z / r) * power;
        __m128d vz2_over_r2 = _mm_div_pd(vz2, r2);
        _mm_storeu_pd(buf_pd, vz2_over_r2);    
        atrig2 = _mm_set_pd(glm::asin(buf_pd[1]), glm::asin(buf_pd[0]));
        __m128d phi2 = _mm_mul_pd(atrig2, power2);

        // dr = power * glm::pow(r, power - 1.) * dr + 1.;
        _mm_storeu_pd(buf_pd, r2);   
        __m128d pow2 = _mm_set_pd(glm::pow(buf_pd[1], power - 1.), glm::pow(buf_pd[0], power - 1.));
        dr2 = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(power2, pow2), dr2), _mm_set1_pd(1.));

        // v = p + glm::pow(r, power) *
        //             vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));  // update vk+1
        _mm_storeu_pd(buf_pd, theta2);
        __m128d cos_theta2 = _mm_set_pd(cos(buf_pd[1]), cos(buf_pd[0]));
        __m128d sin_theta2 = _mm_set_pd(sin(buf_pd[1]), sin(buf_pd[0]));

        _mm_storeu_pd(buf_pd, phi2);
        __m128d cos_phi2 = _mm_set_pd(cos(buf_pd[1]), cos(buf_pd[0]));
        
        __m128d vec3x2 = _mm_mul_pd(cos_theta2, cos_phi2);
        __m128d vec3y2 = _mm_mul_pd(cos_phi2, sin_theta2);
        __m128d vec3z2 = _mm_set_pd(-sin(buf_pd[1]), -sin(buf_pd[0]));

        pow2 = _mm_mul_pd(pow2, r2);
        vx2 = _mm_add_pd(px2, _mm_mul_pd(pow2, vec3x2));
        vy2 = _mm_add_pd(py2, _mm_mul_pd(pow2, vec3y2));
        vz2 = _mm_add_pd(pz2, _mm_mul_pd(pow2, vec3z2));

        // trap = glm::min(trap, r);
        _mm_storeu_pd(buf_pd, r2);
        // trap0 = glm::length(buf_pd[0]);
        // trap1 = glm::length(buf_pd[1]); 
        trap2 = _mm_min_pd(trap2, r2)       

        // r = glm::length(v);    
        vx2_sq = _mm_mul_pd(vx2, vx2)
        vy2_sq = _mm_mul_pd(vy2, vy2)
        vz2_sq = _mm_mul_pd(vz2, vz2)
        r2 = _mm_sqrt_pd(_mm_add_pd(_mm_add_pd(vx2_sq, vy2_sq), vz2_sq));
    

        // if (r > bailout) break; 
        
        _mm_storeu_pd(buf_pd, _mm_cmpgt_pd(r2, bailout2));

        if(buf_pd[1]){
            if (r0_out < bailout){
                r0_out = buf_pd[1];
                if ((r0_out > bailout) && (r1_out > bailout)) break;
            } 
        }

        if(buf_pd[0]){
            if (r1_out < bailout){
                r1_out = buf_pd[0];
                if ((r0_out > bailout) && (r1_out > bailout)) break;
            } 
        }
    }


    // return 0.5 * log(r) * r / dr;  
    r2 = _mm_set_pd(r0_out, r1_out);
    __m128d log_r2 = _mm_set_pd(log(r0_out), log(r1_out));
    out2 = _mm_div_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(0.5), log_r2), r2), dr2);
}

void map2(vec3 p0, vec3 p1, __m128d& trap2, __m128d& out2) {
    vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));

    vec3 rp0 = mat3(1.,   0.,    0.,
                   0., rt.x, -rt.y, 
                   0., rt.y,  rt.x) * p0;  

    vec3 rp1 = mat3(1.,   0.,    0.,
                   0., rt.x, -rt.y, 
                   0., rt.y,  rt.x) * p1;  

    md2(rp0, rp1, trap2, out2);
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


void trace(vec3 ro, vec3 rd0, vec3 rd1, __m128d& trap2, __m128d& d2) {

    double buf_pd[2], len0, len1, t0, t1, t0_out=-1, t1_out=-1;
    
    // double t = 0;   
    __m128d t2 = _mm_set1_pd(0.);

    // double len0 = 0, len0 = 1;  
    __m128d len2 = _mm_set1_pd(0.);
    __m128d ray_multiplier2 = _mm_set1_pd(ray_multiplier);

    for (int i = 0; i < ray_step; ++i) {

        // len = map(ro + rd * t, trap, ID); 
        map2(ro + rd0 * t, ro + rd1 * t, trap2, len2);  
         
        // if (glm::abs(len) < eps || t > far_plane) break;
        _mm_storeu_pd(buf_pd, len2);  
        len0 = buf_pd[1];
        len1 = buf_pd[0];
        _mm_storeu_pd(buf_pd, t2);  
        t0 = buf_pd[1];
        t1 = buf_pd[0];    
        if((t0_out > 0) && (t1_out > 0)) break;
        if (glm::abs(len0) < eps || t0 > far_plane) t0_out = t0;       
        if (glm::abs(len1) < eps || t1 > far_plane) t1_out = t1;       

        // t += len * ray_multiplier;
        t2 = _mm_add_pd(t2, _mm_mul_pd(len2, ray_multiplier2));
    }

    d2 = _mm_set1_pd(t0_out < far_plane ? t0_out : -1.,
                     t1_out < far_plane ? t0_out : -1.);  
}









#define MPI_TAG_JOB_ASSIGN 0
#define MPI_TAG_JOB_CANCEL 1
#define MPI_TAG_TERMINATE 2
#define MPI_TAG_JOB_UPLOAD 3

#define BUF_SIZE_JOB_UPLOAD 4000
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

        // for(int i=0;i<num_threads;i++){
        //     jobTimeTbl[i] = 0;
        // }
   
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

          
                auto start = high_resolution_clock::now();

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
                auto start = high_resolution_clock::now();
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start);
                // cerr<<"[pid "<< rank <<", tid: "<< tid <<"] dt: "<<duration.count()<<" us"<<endl;
                // fprintf(stderr, "[pid %d, tid: %d] fprintf(): dt: %d us\n", rank, tid, (int)duration.count());

                // jobTimeTbl[tid] += (int)duration.count();

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

            auto start = high_resolution_clock::now();

            
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

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            
            // jobTimeTbl[num_threads - 1] += (int)duration.count();
            
    
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



    vec4 partial_AA(int i, int j, int m){

        

        vec2 iResolution = vec2(width, height);

        vec2 p0 = vec2(j, i) + vec2(m, 0.) / (double)AA;
        vec2 p1 = vec2(j, i) + vec2(m, 1.) / (double)AA;

        vec2 uv0 = (-iResolution.xy() + 2. * p0) / iResolution.y;
        vec2 uv1 = (-iResolution.xy() + 2. * p1) / iResolution.y;

        uv0.y *= -1;  
        uv1.y *= -1; 

        vec3 ro = camera_pos;               
        vec3 ta = target_pos;               
        vec3 cf = glm::normalize(ta - ro);  
        vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.))); 
        vec3 cu = glm::normalize(glm::cross(cs, cf));     

        vec3 rd0 = glm::normalize(uv0.x * cs + uv0.y * cu + FOV * cf); 
        vec3 rd1 = glm::normalize(uv1.x * cs + uv1.y * cu + FOV * cf); 
        

        double trap;  


        // double d = trace(ro, rd0, rd1, trap);
        __m128d d2;
        trace(ro, rd0, rd1, trap0, trap1, d2); // dependent loop, 10000 * 24
        

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

    // concurrent_unordered_map<int, int> jobTimeTbl;
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
 
        raw_image = new unsigned char[tmPtr->width * tmPtr->height * 4];
        image = new unsigned char*[tmPtr->height];

        for (int i = 0; i < tmPtr->height; ++i) {
            image[i] = raw_image + i * tmPtr->width * 4;
        }


        for(int i=startIdx; i<=stopIdx; i++){
            dynamicJobQueue.push_back(i);
        }

        init_procCtrlTbl();
        


        // loadDistTbl = new int*[worldSize];
        // for(int i=0;i<worldSize;i++){
        //     loadDistTbl[i] = new int[loadDistLen];
        // }

        // for(int i=0;i<worldSize;i++){
        //     for(int j=0;j<loadDistLen;j++){
        //         loadDistTbl[i][j] = 0;
        //     }
        // }
        
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

    

        if(tmPtr->jobQueue.unsafe_size() < 10 * tmPtr->nThrds){
            

            for(int i=0; i < 10 * tmPtr->nThrds; i++){
                
                jobIdx_t jobIdx;
                if(dynamicJobQueue.try_pop_front(&jobIdx)){
                    Job job;
                    job.idx = jobIdx;

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

                tmPtr->one_job(); // 3000 us.   

                if(tmPtr->jobDoneQueue.unsafe_size() > BUF_SIZE_JOB_UPLOAD / 2){
                    local_receive_completed_jobs(); // 350 us
                }                
            }
            else{
                local_receive_completed_jobs(); // 350 us
                if(check_send_terminate_signal()) return;
            }

            receive_completed_jobs(); // probe: 2us. if true: 250 us
            dynamic_job_schedule();
            dynamic_job_enqueue(); // if true: 24 us.

            // probe_load_distribution();
        }  
    }
    




    // int loadDistLen = 100000;
    // int **loadDistTbl;
    // int loadDist_ofst=0;

    // void probe_load_distribution(){

    //     loadDist_ofst++;
    //     if(loadDist_ofst >= loadDistLen){
    //         fprintf(stdout, \
    //          "[probe_load_distribution()] Warning: loadDist_ofst >= loadDistLen\n");
    //     } 
    //     else{
    //         for(int i=0;i<worldSize;i++){
    //             loadDistTbl[i][loadDist_ofst] = procCtrlTbl[i].jobQueue.size();
    //         }
    //     }
    // }



    // void print_stat(){
        
    //     // ofstream outfile;
    //     // string outfileName = "jobTime" + to_string(rank) + ".txt";        
    //     // outfile.open (outfileName);

    //     // for(int i=0;i<tmPtr->num_threads;i++){
    //     //     outfile << "[print_stat()] jobTimeTbl[" << i << "]: " << tmPtr->jobTimeTbl[i] << "\n"; 
    //     // }
        
    //     // outfile.close();
    //     fprintf(stdout, "[print_stat()] loadDist_ofst: %d\n", loadDist_ofst);

    //     ofstream outfile;
    //     string outfileName;

    //     for(int i=0; i < worldSize; i++){
            
    //         outfileName = "loadDist_pid" + to_string(i) + ".txt";  
    //         outfile.open(outfileName);

    //         for(int j=0; j < loadDist_ofst; j++){
    //             outfile << loadDistTbl[i][j] << "\n"; 
    //         }

    //         outfile.close();
    //     }
    // }



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

            // ofst++;
        }


        // if(ofst > 0){
        //     nJobsCmpltd += ofst;

        //     fprintf(stdout, "[proc %d][local_receive_completed_jobs()] from pid 0 recvSz: %d\n",
        //     rank, nJobsCmpltd);                                     
        // }
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
            
            fprintf(stderr, "[proc %d][receive_completed_jobs()] from pid %d recvSz: %d\n",
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

                
                image[i][4 * j + 0] = (unsigned char)((data >> 24) & dataMask);  
                image[i][4 * j + 1] = (unsigned char)((data >> 16) & dataMask);  
                image[i][4 * j + 2] = (unsigned char)((data >> 8 ) & dataMask);  
                image[i][4 * j + 3] = (unsigned char)255;  

                // if(jobIdx < 0 || jobIdx >= (tmPtr->nTotalJobs)){
                //     fprintf(stderr, "[proc %d][receive_completed_jobs()] job idx out of range.\n", rank);            
                //     exit(1);
                // }
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
            
            int i;
            for(i=0; i < BUF_SIZE_JOB_SCHDL; i++){
                
                if(!pcb_pop_back_job(maxNJobPid, &jobIdx)){break;}
                pcb_push_back_job(minNJobPid, jobIdx);
                
                jobSchdlBuf[i] = jobIdx;
            }
            dynamic_assign_job(minNJobPid, jobSchdlBuf, i);
            dynamic_cancel_job(maxNJobPid, jobSchdlBuf, i);
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


        for(int pid = 0; pid < worldSize; pid++){
            if(procCtrlTbl[pid].jobQueue.size()) return false;
        }

        int sigBuf = 1;
        for(int pid = 1; pid < worldSize; pid++){
            MPI_Send(&sigBuf, 1, MPI_INT, pid, MPI_TAG_TERMINATE, MPI_COMM_WORLD);
        }

        // print_stat();

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

        // static int cnt=startIdx;

        if(tmPtr->jobQueue.unsafe_size() < 10 * tmPtr->nThrds){
            for(int i=0; i < 10 * tmPtr->nThrds; i++){
                
                
         
                jobIdx_t jobIdx;
                if(dynamicJobQueue.try_pop_front(&jobIdx)){
                    Job job;
                    job.idx = jobIdx;

                    // assert(jobIdx == cnt);
                    // assert(cnt <= stopIdx);
                    // cnt++;
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

                tmPtr->one_job();  // 3000 us.               

                while(tmPtr->jobDoneQueue.unsafe_size() >= BUF_SIZE_JOB_UPLOAD / 2){
                    upload_completed_jobs(); // if true: 90 us.
                }
            }
            else{
                upload_completed_jobs(); // if true: 90 us.
                check_receive_terminate_signal();
            }

            dynamic_job_cancel(); // check: 2us. if true: 570 us.
            dynamic_job_receive(); // check: 2us. if true: 410 us.
            dynamic_job_enqueue();  // if true: 24 us.
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
        

        for(int i = 0; i < recvSz; i++){
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
            fprintf(stderr, "\n[check_receive_terminate_signal()][proc %d] \
            receive terminate signal. Terminating...\n\n", rank);
            // print_stat();
            exit(0);
        }
    }


    // void print_stat(){
        
    //     ofstream outfile;
    //     string outfileName = "jobTime" + to_string(rank) + ".txt";        
    //     outfile.open (outfileName);

    //     for(int i=0;i<tmPtr->num_threads;i++){
    //         outfile << "[print_stat()] jobTimeTbl[" << i << "]: " << tmPtr->jobTimeTbl[i] << "\n"; 
    //     }
        
    //     outfile.close();
    // }




    ThreadManager *tmPtr;

    int jobSchdlBuf[BUF_SIZE_JOB_SCHDL];
    int jobDoneBuf[BUF_SIZE_JOB_UPLOAD];

    MapQueue<jobIdx_t> dynamicJobQueue;

    int rank, worldSize;
    int startIdx, stopIdx, nStaticJobs;  
    int nJobsUploaded;  
    
};    





void write_png(ProcessManager* pm){


    unsigned error = lodepng_encode32_file(pm->filename, pm->raw_image, pm->tmPtr->width, pm->tmPtr->height);


    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));

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


        // pm.write_png();
        write_png(&pm);

    }
    else{
        Process proc(argv, rank, worldSize);
        proc.start_process();
        
    }


   

    return 0;
}