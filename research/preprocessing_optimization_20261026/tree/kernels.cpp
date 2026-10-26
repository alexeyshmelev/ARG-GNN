// Exact CPU structural kernels. No floating point approximation or model code.
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <exception>
extern "C" {
// Child insertion order matches Python's ascending children then LIFO stack.
int tree_layout(const int64_t* p, int64_t n, const int64_t* samples, int64_t h,
                int64_t* post, int64_t* depth, int64_t* height, int64_t* desc) {
 try {
  if(n<=0 || n>=int64_t(1)<<31 || h<=0)return 8;
  for(int64_t i=0;i<n;++i)if(p[i]<-1 || p[i]>=n)return 8;
  for(int64_t i=0;i<h;++i)if(samples[i]<0 || samples[i]>=n)return 8;
  std::vector<int64_t> head(n,-1), next(n,-1), stack, preorder;
  int64_t root=-1;
  for(int64_t i=n-1;i>=0;--i){if(p[i]>=0){next[i]=head[p[i]];head[p[i]]=i;}else root=i;}
  for(int64_t i=0;i<h;++i) desc[samples[i]]=1;
  if(root<0 || root>=n)return 8;
  stack.push_back(root);
  while(!stack.empty()) {
   int64_t u=stack.back();stack.pop_back();preorder.push_back(u);
   if(static_cast<int64_t>(preorder.size())>n)return 1;
   for(int64_t c=head[u];c>=0;c=next[c]){depth[c]=depth[u]+1;stack.push_back(c);}
  }
  if(static_cast<int64_t>(preorder.size())!=n)return 1;
  for(int64_t i=0;i<n;++i) {
   int64_t u=preorder[n-1-i];post[i]=u;
   if(p[u]>=0){desc[p[u]]+=desc[u];height[p[u]]=std::max(height[p[u]],height[u]+1);}
  }
  for(int64_t i=0;i<n;++i)if(desc[i]==0)return 2;
  return 0;
 } catch(const std::exception&){return 9;}
}
int forest_index(const int64_t* p,int64_t n,int32_t* head,int32_t* sibling,
                 int32_t* roots,int64_t* steps) {
 try {
  if(n<=0 || n>=int64_t(1)<<31)return 8;
  for(int64_t i=0;i<n;++i)if(p[i]<-1 || p[i]>=n)return 8;
  std::vector<int64_t> trail;
  *steps=0;
  for(int64_t u=0;u<n;++u)if(p[u]>=0){sibling[u]=head[p[u]];head[p[u]]=u;}
  for(int64_t start=0;start<n;++start){
   if(roots[start]!=-2)continue;
   trail.clear();int64_t u=start;
   while(roots[u]==-2 && p[u]>=0){roots[u]=-3;trail.push_back(u);u=p[u];++*steps;}
   int32_t root;
   if(p[u]<0){root=u;roots[u]=root;}else{if(roots[u]<0)return 1;root=roots[u];}
   for(auto it=trail.rbegin();it!=trail.rend();++it)roots[*it]=root;
  }
  return 0;
 } catch(const std::exception&){return 9;}
}
int tree_clades(int64_t root,const int64_t* samples,int64_t h,
                const int32_t* head,const int32_t* sibling,int64_t n,
                int64_t* proper_nodes,uint64_t* proper_masks,int64_t* result) {
 try {
  if(n<=0 || root<0 || root>=n || h<=0 || h>=int64_t(1)<<31)return 8;
  for(int64_t i=0;i<h;++i)if(samples[i]<0 || samples[i]>=n)return 8;
  std::unordered_map<int64_t,int64_t> sampled;
  for(int64_t i=0;i<h;++i)if(!sampled.emplace(samples[i],i).second)return 1;
  std::vector<int64_t> stack{root}, stack_parent{-1}, preorder, parent_index, degree;
  int64_t edges_visited=0;
  while(!stack.empty()) {
   int64_t u=stack.back(),par=stack_parent.back();stack.pop_back();stack_parent.pop_back();
   int64_t local=preorder.size();preorder.push_back(u);parent_index.push_back(par);degree.push_back(0);
   if(static_cast<int64_t>(preorder.size())>n)return 2;
   for(int64_t c=head[u];c>=0;c=sibling[c]){
    if(c>=n || ++edges_visited>n)return 8;
    stack.push_back(c);stack_parent.push_back(local);++degree[local];
   }
  }
  int64_t count=preorder.size(),words=(h+63)/64,leaves=0,members=0,unary=0;
  std::vector<uint64_t> masks(count*words,0);
  for(int64_t i=count-1;i>=0;--i){
   int64_t u=preorder[i];auto found=sampled.find(u);
   if(degree[i]==0){
    if(found==sampled.end())return 3;
    int64_t column=found->second;masks[i*words+column/64] |= uint64_t(1)<<(column%64);++leaves;
   }else if(found!=sampled.end())return 4;
   int64_t size=0;for(int64_t w=0;w<words;++w)size+=__builtin_popcountll(masks[i*words+w]);
   if(size>1 && size<h){
    if(degree[i]==1)++unary;
    else {if(members>=h)return 8;proper_nodes[members]=u;std::copy_n(&masks[i*words],words,proper_masks+members*words);++members;}
   }
   if(parent_index[i]>=0){for(int64_t w=0;w<words;++w)masks[parent_index[i]*words+w]|=masks[i*words+w];}
  }
  if(leaves!=h)return 5;
  for(int64_t w=0;w<words;++w){
   uint64_t expected=(w==words-1 && h%64)?((uint64_t(1)<<(h%64))-1):~uint64_t(0);
   if(masks[w]!=expected)return 5;
  }
  result[0]=members;result[1]=count;result[2]=unary;return 0;
 }catch(const std::exception&){return 9;}
}
}
extern "C" {
int cut_index(const int64_t* p,const float* times,int64_t n,
              const int64_t* roots,int64_t root_count,
              int64_t* head,int64_t* sibling,int32_t* owner,int64_t* cut) {
 try {
  if(n<=0 || n>=int64_t(1)<<31 || root_count<=0 || root_count>n)return 8;
  for(int64_t i=0;i<n;++i)if(p[i]<-1 || p[i]>=n)return 8;
  for(int64_t i=0;i<root_count;++i)if(roots[i]<0 || roots[i]>=n)return 8;
  for(int64_t c=0;c<n;++c)if(p[c]>=0){sibling[c]=head[p[c]];head[p[c]]=c;}
  std::vector<int64_t> stack;
  for(int64_t tree_id=0;tree_id<root_count;++tree_id){
   stack.push_back(roots[tree_id]);
   while(!stack.empty()){
    int64_t u=stack.back();stack.pop_back();
    if(owner[u]>=0)return 1;
    owner[u]=tree_id;
    if(times[u]<=350.0f){cut[u]=(p[u]<0 || times[p[u]]>350.0f)?u:cut[p[u]];if(cut[u]<0)return 2;}
    for(int64_t c=head[u];c>=0;c=sibling[c])stack.push_back(c);
   }
  }
  for(int64_t i=0;i<n;++i)if(owner[i]<0)return 3;
  return 0;
 }catch(const std::exception&){return 9;}
}
int cut_groups(const int64_t* p,const float* times,const int64_t* cut,int64_t n,
               const int64_t* samples,const uint8_t* above,int64_t t,int64_t h,
               int64_t* local,int64_t* global,int64_t* reps,int64_t* counts,
               uint8_t* root_flags,int64_t* offsets) {
 try {
  if(n<=0 || t<=0 || h<=0)return 8;
  int64_t total=0;offsets[0]=0;
  for(int64_t at=0;at<t;++at){
   std::unordered_map<int64_t,int64_t> lookup;
   for(int64_t i=0;i<h;++i){
    int64_t u=samples[at*h+i];if(u<0 || u>=n)return 8;
    int64_t r=above[i]?u:cut[u];if(r<0 || r>=n)return 8;
    auto found=lookup.find(r);int64_t k;
    if(found==lookup.end()){
     k=lookup.size();lookup.emplace(r,k);reps[total+k]=r;
     root_flags[total+k]=(p[r]<0 && times[r]<=350.0f);counts[total+k]=0;
    }else k=found->second;
    local[at*h+i]=k;global[at*h+i]=total+k;++counts[total+k];
   }
   total+=lookup.size();offsets[at+1]=total;
  }
  return 0;
 }catch(const std::exception&){return 9;}
}
int direction_layouts(const int64_t* groups,int64_t g,int64_t h,int reverse,
                       int64_t* cells,int64_t* current,int64_t* old,float* counts) {
 try {
  if(g<=0 || h<=0 || h>=int64_t(1)<<31)return 8;
  for(int64_t i=0;i<g*h;++i)if(groups[i]<0 || groups[i]>=h)return 8;
  int64_t previous=-1;
  for(int64_t step=0;step<g;++step){
   int64_t e=reverse?g-1-step:step;
   std::vector<int64_t> exact_counts(h,0);
   if(previous<0){
    int64_t maximum=0;
    for(int64_t i=0;i<h;++i){int64_t k=groups[e*h+i];cells[e*h+i]=k;++exact_counts[k];maximum=std::max(maximum,k);}
    for(int64_t k=0;k<=maximum;++k){current[e*h+k]=k;old[e*h+k]=k;}
   }else{
    std::unordered_map<int64_t,int64_t> lookup;
    for(int64_t i=0;i<h;++i){
     int64_t a=groups[previous*h+i],b=groups[e*h+i],key=a*h+b,k;
     auto found=lookup.find(key);
     if(found==lookup.end()){k=lookup.size();lookup.emplace(key,k);old[e*h+k]=a;current[e*h+k]=b;}else k=found->second;
     cells[e*h+i]=k;++exact_counts[k];
    }
   }
   for(int64_t k=0;k<h;++k)counts[e*h+k]=static_cast<float>(exact_counts[k]);
   previous=e;
  }
  return 0;
 }catch(const std::exception&){return 9;}
}
}
