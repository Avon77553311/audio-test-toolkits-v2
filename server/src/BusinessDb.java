import com.google.gson.*;
import java.nio.file.*;
import java.sql.*;
import java.util.*;

/** Local transactional business storage. Algorithm job databases remain private to Python. */
final class BusinessDb {
    static final Gson JSON = new GsonBuilder().disableHtmlEscaping().create();
    final String url;
    interface Work<T> { T run(Connection db) throws Exception; }
    BusinessDb(Path runtime) throws Exception {
        Files.createDirectories(runtime);
        Class.forName("org.sqlite.JDBC");
        url="jdbc:sqlite:"+runtime.resolve("business.sqlite3").toAbsolutePath();
        try(Connection db=open(); Statement s=db.createStatement()) {
            try(var version=s.executeQuery("PRAGMA user_version")){if(version.next()&&version.getInt(1)>1)throw new SQLException("Business database requires a newer application; refusing incompatible downgrade");}
            s.execute("PRAGMA journal_mode=WAL");
            for(String sql : List.of(
                "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,login TEXT NOT NULL UNIQUE,name TEXT NOT NULL,password TEXT,google_sub TEXT UNIQUE,role TEXT NOT NULL DEFAULT 'user',status TEXT NOT NULL DEFAULT 'active',balance INTEGER NOT NULL DEFAULT 0 CHECK(balance>=0),quota INTEGER NOT NULL DEFAULT 5 CHECK(quota>=0),created TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),csrf TEXT NOT NULL,expires INTEGER NOT NULL)",
                "CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),submission TEXT NOT NULL,request_hash TEXT NOT NULL DEFAULT '',tool TEXT NOT NULL DEFAULT '',resource TEXT NOT NULL DEFAULT 'cpu',status TEXT NOT NULL DEFAULT 'uploading',paid INTEGER NOT NULL DEFAULT 0,price INTEGER NOT NULL DEFAULT 0,free INTEGER NOT NULL DEFAULT 0,month TEXT NOT NULL DEFAULT '',created TEXT NOT NULL,updated TEXT NOT NULL,result TEXT,UNIQUE(user_id,submission))",
                "CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_user ON tasks(user_id) WHERE status IN ('uploading','queued','running')",
                "CREATE TABLE IF NOT EXISTS usage(task_id TEXT PRIMARY KEY REFERENCES tasks(id),user_id TEXT NOT NULL,month TEXT NOT NULL,amount INTEGER NOT NULL,free INTEGER NOT NULL,created TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS credits(id TEXT PRIMARY KEY,user_id TEXT NOT NULL,actor TEXT NOT NULL,amount INTEGER NOT NULL,reason TEXT NOT NULL,created TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT,actor TEXT NOT NULL,action TEXT NOT NULL,target TEXT NOT NULL,detail TEXT NOT NULL,created TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS oauth_states(state TEXT PRIMARY KEY,verifier TEXT NOT NULL,expires INTEGER NOT NULL)"
            )) s.execute(sql);
            s.execute("INSERT OR IGNORE INTO settings VALUES('cpu_concurrency','2'),('gpu_concurrency','2')");
            s.execute("PRAGMA user_version=1");
        }
    }
    Connection open() throws SQLException {
        Connection c=DriverManager.getConnection(url);
        try(Statement s=c.createStatement()){s.execute("PRAGMA busy_timeout=15000");s.execute("PRAGMA foreign_keys=ON");}
        return c;
    }
    synchronized <T> T tx(Work<T> action) throws Exception {
        try(Connection db=open();Statement s=db.createStatement()) {
            s.execute("BEGIN IMMEDIATE");
            try { T value=action.run(db);s.execute("COMMIT");return value; }
            catch(Exception error){s.execute("ROLLBACK");throw error;}
        }
    }
    <T> T read(Work<T> action) throws Exception {try(Connection db=open()){return action.run(db);}}
    static PreparedStatement statement(Connection db,String sql,Object...args) throws SQLException {
        PreparedStatement p=db.prepareStatement(sql);
        for(int i=0;i<args.length;i++)p.setObject(i+1,args[i]);
        return p;
    }
    static int update(Connection db,String sql,Object...args) throws SQLException {try(var p=statement(db,sql,args)){return p.executeUpdate();}}
    static List<Map<String,Object>> rows(Connection db,String sql,Object...args) throws SQLException {
        try(var p=statement(db,sql,args);var r=p.executeQuery()) {
            List<Map<String,Object>> values=new ArrayList<>();var meta=r.getMetaData();
            while(r.next()){Map<String,Object> row=new LinkedHashMap<>();for(int i=1;i<=meta.getColumnCount();i++)row.put(meta.getColumnLabel(i),r.getObject(i));values.add(row);}
            return values;
        }
    }
    static Map<String,Object> one(Connection db,String sql,Object...args) throws SQLException {var rows=rows(db,sql,args);return rows.isEmpty()?null:rows.get(0);}
    static long number(Map<String,Object> row,String key) {return ((Number)row.get(key)).longValue();}
    static String text(Map<String,Object> row,String key) {Object v=row.get(key);return v==null?"":v.toString();}
    static String now(){return java.time.Instant.now().toString();}
    static void audit(Connection db,String actor,String action,String target,Object detail) throws SQLException {update(db,"INSERT INTO audit(actor,action,target,detail,created) VALUES(?,?,?,?,?)",actor,action,target,JSON.toJson(detail),now());}
}
