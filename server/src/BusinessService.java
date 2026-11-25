import com.sun.net.httpserver.HttpExchange;
import com.google.gson.*;
import java.nio.file.Path;
import java.sql.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;

/** User ownership, durable dispatch, atomic quota settlement and administrator APIs. */
final class BusinessService implements AutoCloseable {
    final BusinessDb db;final BusinessAuth auth;final AlgorithmService worker;
    final ScheduledExecutorService scheduler=Executors.newSingleThreadScheduledExecutor(r->{Thread t=new Thread(r,"business-dispatch");t.setDaemon(true);return t;});
    final Set<String> active=Set.of("uploading","queued","running");
    final Set<String> rendering=ConcurrentHashMap.newKeySet();
    final java.util.concurrent.atomic.AtomicInteger renderCount=new java.util.concurrent.atomic.AtomicInteger();
    BusinessService(Path project,AlgorithmService worker)throws Exception {
        this.worker=worker;db=new BusinessDb(project.resolve("runtime"));auth=new BusinessAuth(db,project);
        scheduler.scheduleWithFixedDelay(()->{try{tick();}catch(Exception e){System.err.println("Business reconcile: "+e.getClass().getSimpleName());}},1,1,TimeUnit.SECONDS);
    }
    static Map<String,Object> row(Connection c,String sql,Object...args)throws Exception{return BusinessDb.one(c,sql,args);}
    static List<Map<String,Object>> rows(Connection c,String sql,Object...args)throws Exception{return BusinessDb.rows(c,sql,args);}
    static int put(Connection c,String sql,Object...args)throws Exception{return BusinessDb.update(c,sql,args);}
    static String str(Map<String,Object> r,String k){return BusinessDb.text(r,k);}
    static long num(Map<String,Object> r,String k){return BusinessDb.number(r,k);}
    static String now(){return BusinessDb.now();}
    static String month(){return LocalDate.now(ZoneId.of("Asia/Shanghai")).withDayOfMonth(1).toString();}
    static BusinessAuth.Failure fail(int status,String text){return new BusinessAuth.Failure(status,text);}
    static void reply(HttpExchange x,int status,Object v)throws Exception{BusinessAuth.reply(x,status,v);}
    static JsonObject json(byte[] b){return JsonParser.parseString(new String(b,java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject();}
    JsonObject call(String method,String path,String body)throws Exception {
        var r=worker.call(method,path,body);var value=json(r.body());
        if(r.statusCode()>=400)throw fail(r.statusCode(),value.has("message")?value.get("message").getAsString():"算法服务请求失败");return value;
    }
    int concurrency(Connection c,String resource)throws Exception{return Integer.parseInt(str(row(c,"SELECT value FROM settings WHERE key=?",resource+"_concurrency"),"value"));}
    long price(String tool) {
        String key=switch(tool){case "stems"->"STEMS";case "voice"->"VOCAL_TRANSFORM";case "song"->"SONG_INFO";case "chords"->"CHORD_ANALYSIS";case "midi"->"AUDIO_TO_MIDI";case "speech"->"SPEECH";default->"";};
        long fallback=switch(tool){case "voice"->150;case "song"->50;case "speech"->0;default->100;};
        try{long p=Long.parseLong(auth.value("GPU_PRICE_"+key+"_MINOR",Long.toString(fallback)));return p>0&&p<=100000000?p:fallback;}catch(NumberFormatException e){return fallback;}
    }
    Map<String,Object> own(BusinessAuth.Identity user,String id)throws Exception {
        if(!id.matches("[0-9a-f]{32}"))throw fail(404,"任务不存在");
        var task=db.read(c->row(c,"SELECT * FROM tasks WHERE id=? AND user_id=?",id,user.id()));if(task==null)throw fail(404,"任务不存在");return task;
    }
    JsonObject publicTask(Map<String,Object> r) {
        JsonObject t=str(r,"result").isEmpty()?new JsonObject():JsonParser.parseString(str(r,"result")).getAsJsonObject();
        t.addProperty("id",str(r,"id"));t.addProperty("tool",str(r,"tool"));t.addProperty("status",str(r,"status"));t.addProperty("created",str(r,"created"));t.addProperty("updated",str(r,"updated"));
        if(!t.has("name"))t.addProperty("name","音频任务");if(!t.has("progress"))t.addProperty("progress",0);
        if(!t.has("message"))t.addProperty("message",str(r,"status").equals("uploading")?"正在接收音频":"等待处理");
        t.addProperty("cancelable",active.contains(str(r,"status"))&&!(str(r,"status").equals("running")&&num(r,"paid")!=0));
        t.addProperty("paid",num(r,"paid")!=0);t.addProperty("resource",str(r,"resource"));return t;
    }
    void synchronizeTask(String id,JsonObject task)throws Exception {
        String status=task.get("status").getAsString();if(!Set.of("queued","running","completed","failed","cancelled").contains(status))throw fail(502,"算法任务状态无效");
        db.tx(c->{var previous=row(c,"SELECT * FROM tasks WHERE id=?",id);if(previous==null)return null;
            if(Set.of("completed","failed","cancelled").contains(str(previous,"status")))return null;
            // Preserve a Java queue until the Java dispatcher has admitted execution.
            if(str(previous,"status").equals("queued")&&status.equals("running"))return null;
            put(c,"UPDATE tasks SET status=?,updated=?,result=? WHERE id=?",status,now(),task.toString(),id);
            if(status.equals("completed")&&num(previous,"paid")!=0) {
                if(row(c,"SELECT task_id FROM usage WHERE task_id=?",id)==null){
                    long amount=num(previous,"price");
                    if(amount>0&&put(c,"UPDATE users SET balance=balance-? WHERE id=? AND balance>=?",amount,str(previous,"user_id"),amount)!=1)throw fail(409,"结算余额异常，任务保留等待对账");
                    put(c,"INSERT INTO usage VALUES(?,?,?,?,?,?)",id,str(previous,"user_id"),str(previous,"month"),amount,num(previous,"free"),now());
                }
            }
            return null;
        });
    }
    synchronized void tick()throws Exception {
        for(var t:db.read(c->rows(c,"SELECT * FROM tasks WHERE status='running'"))) {
            try {synchronizeTask(str(t,"id"),call("GET","/api/tasks/"+str(t,"id"),null).getAsJsonObject("task"));}
            catch(BusinessAuth.Failure e){if(e.status==404)markFailure(str(t,"id"),"算法任务已丢失，请重新提交");}
        }
        for(var t:db.read(c->rows(c,"SELECT * FROM tasks WHERE status='uploading' AND created<?",Instant.now().minusSeconds(180).toString()))) {
            markFailure(str(t,"id"),"上传未完成，未执行算法或扣费，请重新提交");
        }
        for(String resource:List.of("cpu","gpu")) {
            List<Map<String,Object>> queue=db.read(c->{long count=num(row(c,"SELECT COUNT(*) n FROM tasks WHERE resource=? AND status='running'",resource),"n");return rows(c,"SELECT t.* FROM tasks t JOIN users u ON u.id=t.user_id WHERE t.status='queued' AND t.resource=? AND u.status='active' ORDER BY t.created LIMIT ?",resource,Math.max(0,concurrency(c,resource)-count));});
            for(var t:queue) {
                String id=str(t,"id");boolean claimed=db.tx(c->put(c,"UPDATE tasks SET status='running',updated=? WHERE id=? AND status='queued'",now(),id)==1);
                if(!claimed)continue;
                try{synchronizeTask(id,call("POST","/internal/tasks/"+id+"/start","{}").getAsJsonObject("task"));}
                catch(Exception e){
                    // Unknown remote submission state stays running for reconciliation; never blindly resubmit.
                    if(e instanceof BusinessAuth.Failure f && f.status==404)markFailure(id,"准备的算法任务不存在");
                }
            }
        }
    }
    void markFailure(String id,String message)throws Exception {
        JsonObject info=new JsonObject();info.addProperty("status","failed");info.addProperty("message",message);info.addProperty("progress",0);
        synchronizeTask(id,info);
    }
    void cancel(String id,BusinessAuth.Identity actor,boolean admin)throws Exception {
        var task=db.read(c->row(c,"SELECT * FROM tasks WHERE id=?",id));if(task==null)throw fail(404,"任务不存在");
        if(!admin&&!str(task,"user_id").equals(actor.id()))throw fail(404,"任务不存在");
        if(!active.contains(str(task,"status")))throw fail(409,"任务已经结束");
        if(num(task,"paid")!=0&&str(task,"status").equals("running"))throw fail(409,"付费任务已提交，不能承诺停止远端消费");
        if(str(task,"status").equals("uploading"))throw fail(409,"上传正在进行，请等待上传结束");
        JsonObject result=call("DELETE","/api/tasks/"+id,null).getAsJsonObject("task");synchronizeTask(id,result);
        if(admin)db.tx(c->{BusinessDb.audit(c,actor.id(),"cancel-task",id,Map.of());return null;});
    }
    void createTask(HttpExchange x,BusinessAuth.Identity user,String parent)throws Exception {
        String submission=Objects.toString(x.getRequestHeaders().getFirst("X-Submission-ID"),"");
        if(!submission.matches("[0-9a-fA-F-]{36}"))throw fail(400,"请提供有效的 X-Submission-ID UUID，避免重复提交");
        if(parent!=null)own(user,parent);
        String id=UUID.randomUUID().toString().replace("-","");
        var existing=db.tx(c->{var prior=row(c,"SELECT * FROM tasks WHERE user_id=? AND submission=?",user.id(),submission);if(prior!=null)return prior;
            var maintenance=row(c,"SELECT value FROM settings WHERE key='maintenance'");
            if(maintenance!=null&&str(maintenance,"value").equals("1"))throw fail(503,"工作室正在更新，请稍后提交");
            if(row(c,"SELECT id FROM tasks WHERE user_id=? AND status IN ('uploading','queued','running')",user.id())!=null)throw fail(409,"每个用户同时只能有一个未完成任务");
            if(user.guest()&&num(row(c,"SELECT COUNT(*) n FROM tasks t JOIN users u ON u.id=t.user_id WHERE u.role='guest' AND t.status IN ('uploading','queued','running')"),"n")>=32)throw fail(429,"局域网任务队列已满，请稍后提交");
            put(c,"INSERT INTO tasks(id,user_id,submission,created,updated) VALUES(?,?,?,?,?)",id,user.id(),submission,now(),now());return null;
        });
        if(existing!=null){reply(x,200,Map.of("success",true,"replayed",true,"task",publicTask(existing)));return;}
        boolean admitted=false;
        try {
            var response=worker.capture(x,Map.of("X-Studio-Prepare","true","X-Studio-Task-Id",id));
            var result=json(response.body());if(response.statusCode()>=400)throw fail(response.statusCode(),result.has("message")?result.get("message").getAsString():"音频参数无效");
            JsonObject task=result.getAsJsonObject("task"),intent=call("GET","/internal/tasks/"+id+"/intent",null);
            boolean paid=intent.get("paid").getAsBoolean();String resource=intent.get("resource").getAsString(),tool=intent.get("tool").getAsString();
            db.tx(c->{var account=row(c,"SELECT * FROM users WHERE id=?",user.id());if(!str(account,"status").equals("active"))throw fail(403,"账户不可用");
                long cost=0,free=0;
                if(paid){
                    if(user.guest())throw fail(403,"匿名访客仅开放免费本地处理，云端付费功能仅供管理员使用");
                    if(!auth.enabled("STUDIO_ENABLE_PAID"))throw fail(403,"付费服务未启用");
                    if(!user.admin()) {
                        if(resource.equals("gpu")) {
                            long used=num(row(c,"SELECT COUNT(*) n FROM usage WHERE user_id=? AND month=? AND free=1",user.id(),month()),"n");
                            if(used<num(account,"quota"))free=1;else cost=price(tool);
                        } else {cost=price("speech");if(cost==0)throw fail(503,"付费语音尚未设置服务端价格");}
                        if(num(account,"balance")<cost)throw fail(402,"余额不足，请联系管理员充值");
                    }
                }
                put(c,"UPDATE tasks SET tool=?,resource=?,paid=?,price=?,free=?,month=?,status='queued',updated=?,result=? WHERE id=?",tool,resource,paid?1:0,cost,free,month(),now(),task.toString(),id);return null;
            });
            admitted=true;
            reply(x,202,Map.of("success",true,"task",publicTask(db.read(c->row(c,"SELECT * FROM tasks WHERE id=?",id)))));
        } catch(Exception e) {
            // A broken browser connection must not cancel an already admitted task.
            if(admitted)throw e;
            try{call("DELETE","/api/tasks/"+id,null);}catch(Exception ignored){}
            markFailure(id,e instanceof BusinessAuth.Failure?e.getMessage():"上传失败或结果不明，未启动远端任务，请查看任务状态");throw e;
        }
    }
    boolean handle(HttpExchange x)throws Exception {
        String path=x.getRequestURI().getPath(),method=x.getRequestMethod();
        if(path.startsWith("/api/studio/")||path.startsWith("/api/piano/")||path.matches("/api/tasks/[^/]+/edit"))throw fail(404,"简化版不提供此工具");
        if(auth.handle(x,path))return true;
        if(path.equals("/api/health")){worker.proxy(x);return true;}
        if(path.equals("/api/capabilities")&&method.equals("GET")){
            var data=call("GET",path,null);var login=auth.identity(x);
            if(auth.publicLan()&&(login==null||!login.admin())){
                var features=data.getAsJsonObject("features");for(String key:List.of("paid_enabled","cloud_stems","cloud_chords"))features.addProperty(key,false);
            }
            reply(x,200,data);return true;
        }
        if(!path.startsWith("/api/"))return false;
        boolean management=path.startsWith("/api/admin/")||path.startsWith("/api/account")||path.startsWith("/api/billing/");
        var user=management?auth.require(x):auth.requireTool(x);
        if(management&&auth.publicLan()&&!user.admin())throw fail(403,"需要管理员权限");
        if(path.equals("/api/account")&&method.equals("GET")) {
            reply(x,200,db.read(c->{var account=row(c,"SELECT id,login,name,role,status,balance,quota,created FROM users WHERE id=?",user.id());long used=num(row(c,"SELECT COUNT(*) n FROM usage WHERE user_id=? AND month=? AND free=1",user.id(),month()),"n");return Map.of("success",true,"account",account,"month",month(),"quotaUsed",used,"currency","CNY");}));return true;
        }
        if(path.equals("/api/billing/usage")&&method.equals("GET")){int page=page(x);reply(x,200,Map.of("success",true,"usage",db.read(c->rows(c,"SELECT * FROM usage WHERE user_id=? ORDER BY created DESC LIMIT 30 OFFSET ?",user.id(),page*30))));return true;}
        if(path.equals("/api/account/password")&&method.equals("POST")) {
            var b=BusinessAuth.body(x);String old=BusinessAuth.text(b,"oldPassword"),hash=BusinessAuth.password(BusinessAuth.text(b,"password"));
            db.tx(c->{var a=row(c,"SELECT password FROM users WHERE id=?",user.id());String[] parts=str(a,"password").split(":",2);if(parts.length!=2||!BusinessAuth.equal(parts[1],BusinessAuth.derive(old,parts[0])))throw fail(403,"原密码不正确");put(c,"UPDATE users SET password=? WHERE id=?",hash,user.id());put(c,"DELETE FROM sessions WHERE user_id=? AND token<>?",user.id(),user.session());BusinessDb.audit(c,user.id(),"password-change",user.id(),Map.of());return null;});reply(x,200,Map.of("success",true));return true;
        }
        if(path.equals("/api/account/delete")&&method.equals("POST")) {
            db.tx(c->{if(user.admin())throw fail(409,"请先由另一位管理员调整账号权限");if(row(c,"SELECT id FROM tasks WHERE user_id=? AND status IN ('uploading','queued','running')",user.id())!=null)throw fail(409,"仍有未完成任务，暂不可注销");put(c,"UPDATE users SET status='deleted' WHERE id=?",user.id());put(c,"DELETE FROM sessions WHERE user_id=?",user.id());BusinessDb.audit(c,user.id(),"delete-account",user.id(),Map.of());return null;});auth.setCookie(x,"studio_session","",0,"/");reply(x,200,Map.of("success",true));return true;
        }
        if(path.startsWith("/api/admin/")){if(!user.admin())throw fail(403,"需要管理员权限");return admin(x,user,path,method);}
        if(path.equals("/api/tasks")&&method.equals("GET")){reply(x,200,Map.of("success",true,"tasks",db.read(c->rows(c,"SELECT * FROM tasks WHERE user_id=? ORDER BY created DESC LIMIT 100",user.id())).stream().map(this::publicTask).toList()));return true;}
        if(path.equals("/api/tasks/active")&&method.equals("GET")){reply(x,200,Map.of("success",true,"tasks",db.read(c->rows(c,"SELECT * FROM tasks WHERE user_id=? AND status IN ('uploading','queued','running')",user.id())).stream().map(this::publicTask).toList()));return true;}
        if(path.equals("/api/tasks")&&method.equals("POST")){createTask(x,user,null);return true;}
        if(path.matches("/api/tasks/[0-9a-f]{32}")) {
            String id=path.split("/")[3];var t=own(user,id);
            if(method.equals("DELETE")){cancel(id,user,false);t=own(user,id);}
            else if(!method.equals("GET"))throw fail(405,"不支持的请求方法");
            reply(x,200,Map.of("success",true,"task",publicTask(t)));return true;
        }
        if(path.startsWith("/api/files/")) {String[] p=path.split("/");if(p.length!=5)throw fail(404,"文件不存在");own(user,p[3]);worker.proxy(x);return true;}
        throw fail(404,"接口不存在");
    }
    static int page(HttpExchange x){try{return Math.max(0,Math.min(10000,Integer.parseInt(BusinessAuth.query(x).getOrDefault("page","0"))));}catch(NumberFormatException e){throw fail(400,"分页参数无效");}}
    static long integer(JsonObject b,String key,long min,long max) {
        try {var v=b.get(key);if(v==null||!v.isJsonPrimitive()||!v.getAsJsonPrimitive().isNumber())throw new IllegalArgumentException();long n=v.getAsBigDecimal().longValueExact();if(n<min||n>max)throw new IllegalArgumentException();return n;}catch(Exception e){throw fail(400,key+" 必须为指定范围的整数");}
    }
    static String reason(JsonObject b){String r=BusinessAuth.text(b,"reason").strip();if(r.length()<2||r.length()>300)throw fail(400,"请填写 2–300 字操作理由");return r;}
    boolean admin(HttpExchange x,BusinessAuth.Identity actor,String path,String method)throws Exception {
        int page=page(x);
        if(path.equals("/api/admin/users")&&method.equals("GET")){String search=BusinessAuth.query(x).getOrDefault("q","");reply(x,200,Map.of("success",true,"users",db.read(c->rows(c,"SELECT id,login,name,role,status,balance,quota,created FROM users WHERE login LIKE ? OR name LIKE ? ORDER BY created DESC LIMIT 30 OFFSET ?","%"+search+"%","%"+search+"%",page*30))));return true;}
        if(path.equals("/api/admin/users")&&method.equals("POST")) {
            var b=BusinessAuth.body(x);String login=BusinessAuth.login(BusinessAuth.text(b,"login")),name=BusinessAuth.text(b,"name").strip(),hash=BusinessAuth.password(BusinessAuth.text(b,"password")),id=UUID.randomUUID().toString();
            db.tx(c->{if(row(c,"SELECT id FROM users WHERE login=?",login)!=null)throw fail(409,"账号已存在");put(c,"INSERT INTO users(id,login,name,password,created) VALUES(?,?,?,?,?)",id,login,name.isBlank()?login:name.substring(0,Math.min(name.length(),80)),hash,now());BusinessDb.audit(c,actor.id(),"create-user",id,Map.of("login",login));return null;});reply(x,201,Map.of("success",true,"id",id));return true;
        }
        if(path.matches("/api/admin/users/[a-f0-9-]{36}/(status|credit|quota|role)")&&Set.of("POST","PATCH").contains(method)) {
            String[] parts=path.split("/");String id=parts[4],action=parts[5];var b=BusinessAuth.body(x);String why=reason(b);
            db.tx(c->{var target=row(c,"SELECT * FROM users WHERE id=?",id);if(target==null)throw fail(404,"用户不存在");
                if(action.equals("credit")) {
                    long amount=integer(b,"amountMinor",1,100000000);String request=BusinessAuth.text(b,"requestId");if(!request.matches("[0-9a-fA-F-]{36}"))throw fail(400,"充值需要唯一 requestId UUID");
                    var prior=row(c,"SELECT * FROM credits WHERE id=?",request);if(prior!=null){if(!str(prior,"user_id").equals(id)||num(prior,"amount")!=amount)throw fail(409,"充值请求编号已用于其他操作");return null;}
                    if(str(target,"status").equals("deleted"))throw fail(409,"账户已注销");put(c,"UPDATE users SET balance=balance+? WHERE id=?",amount,id);put(c,"INSERT INTO credits VALUES(?,?,?,?,?,?)",request,id,actor.id(),amount,why,now());
                } else if(action.equals("quota")){long quota=integer(b,"quota",0,1000000);long used=num(row(c,"SELECT COUNT(*) n FROM usage WHERE user_id=? AND month=? AND free=1",id,month()),"n");if(quota<used)throw fail(409,"配额不能小于当月已用次数");if(row(c,"SELECT id FROM tasks WHERE user_id=? AND status IN ('uploading','queued','running')",id)!=null)throw fail(409,"请在该用户任务结束后调整配额");put(c,"UPDATE users SET quota=? WHERE id=?",quota,id);}
                else {
                    String value=BusinessAuth.text(b,action);Set<String> valid=action.equals("role")?Set.of("user","admin"):Set.of("active","blocked");if(!valid.contains(value))throw fail(400,"状态或角色无效");
                    if(str(target,"status").equals("deleted"))throw fail(409,"注销账户不可恢复");
                    boolean removeAdmin=str(target,"role").equals("admin")&&(action.equals("role")&&value.equals("user")||action.equals("status")&&value.equals("blocked"));
                    if(removeAdmin&&num(row(c,"SELECT COUNT(*) n FROM users WHERE role='admin' AND status='active'"),"n")<=1)throw fail(409,"必须保留至少一个有效管理员");
                    put(c,"UPDATE users SET "+action+"=? WHERE id=?",value,id);put(c,"DELETE FROM sessions WHERE user_id=?",id);
                }
                BusinessDb.audit(c,actor.id(),"user-"+action,id,Map.of("reason",why));return null;
            });reply(x,200,Map.of("success",true));return true;
        }
        if(path.equals("/api/admin/tasks")&&method.equals("GET")){reply(x,200,Map.of("success",true,"tasks",db.read(c->rows(c,"SELECT t.*,u.login FROM tasks t JOIN users u ON u.id=t.user_id ORDER BY t.created DESC LIMIT 50 OFFSET ?",page*50)).stream().map(r->{var out=publicTask(r);out.addProperty("user",str(r,"login"));return out;}).toList()));return true;}
        if(path.matches("/api/admin/tasks/[0-9a-f]{32}/cancel")&&method.equals("POST")){cancel(path.split("/")[4],actor,true);reply(x,200,Map.of("success",true));return true;}
        if(path.equals("/api/admin/gpu-usage")&&method.equals("GET")){reply(x,200,Map.of("success",true,"usage",db.read(c->rows(c,"SELECT g.*,u.login FROM usage g JOIN users u ON u.id=g.user_id ORDER BY g.created DESC LIMIT 50 OFFSET ?",page*50))));return true;}
        if(path.equals("/api/admin/audit-logs")&&method.equals("GET")){reply(x,200,Map.of("success",true,"logs",db.read(c->rows(c,"SELECT a.*,u.login FROM audit a LEFT JOIN users u ON u.id=a.actor ORDER BY a.id DESC LIMIT 50 OFFSET ?",page*50))));return true;}
        if(path.equals("/api/admin/concurrency")) {
            if(method.equals("PATCH")){var b=BusinessAuth.body(x);long cpu=integer(b,"cpu",1,16),gpu=integer(b,"gpu",1,16);String why=reason(b);db.tx(c->{put(c,"UPDATE settings SET value=? WHERE key='cpu_concurrency'",Long.toString(cpu));put(c,"UPDATE settings SET value=? WHERE key='gpu_concurrency'",Long.toString(gpu));BusinessDb.audit(c,actor.id(),"concurrency","global",Map.of("cpu",cpu,"gpu",gpu,"reason",why));return null;});}
            else if(!method.equals("GET"))throw fail(405,"请求方法无效");
            reply(x,200,db.read(c->Map.of("success",true,"cpu",concurrency(c,"cpu"),"gpu",concurrency(c,"gpu"))));return true;
        }
        throw fail(404,"管理接口不存在");
    }
    @Override public void close(){scheduler.shutdownNow();}
}
