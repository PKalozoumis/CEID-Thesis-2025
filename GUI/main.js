const {app, BrowserWindow, session, ipcMain} = require("electron");
const Database = require("better-sqlite3")
const path = require("path")
const fs = require("fs");

require('electron-reload')(__dirname, {
    electron: require('path').join(__dirname, 'node_modules', '.bin', 'electron')
});

//Setup database
const db = new Database(path.join(app.getPath('userData'), "db.sqlite"))
//const db = new Database("temp/db.sqlite")
db.prepare("CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY,query TEXT,time TEXT,response TEXT)").run()

//======================================================================================

function isURL(url)
{
    try
    {
        new URL(url);
        return true;
    }
    catch(_)
    {
        return false;
    }
}

//======================================================================================

const createWindow = (source, options = null)=>{
    const defaultOptions = {
        width: 1280,
        height: 720,
        minWidth: 320,
        minHeight: 180,
        show:false,
        webPreferences: {
            nodeIntegration: true,
            preload: path.join(__dirname, "preload.js")
        }
    }

    const win = new BrowserWindow({...defaultOptions, ...options});
    win.maximize()
    win.show()
    win.setMenuBarVisibility(false)
    win.loadFile(source);   
}

//======================================================================================

function setupApiHandlers()
{
    //Write file
    ipcMain.handle('write-file', async (_, filePath, data)=>
    {
        fs.writeFileSync(filePath, data);
    })

    //Add to history
    ipcMain.handle('add-to-history', (_, historyEntry)=>
    {
        let {query, time, response} = historyEntry;
        let info = db.prepare("INSERT INTO history VALUES(?, ?, ?, ?)").run(null, query, time, response);

        console.log("stored")
        return info.lastInsertRowid + 1
    })

    //Load history
    ipcMain.handle('load-history', (_)=>
    {
        let rows = db.prepare("SELECT * FROM history ORDER BY id DESC").all();
        return rows
    })

    //Delete from history
    ipcMain.handle('delete-from-history', (_, id)=>
    {
        console.log(`Deleted ${id}`)
        db.prepare("DELETE FROM history WHERE id=?").run(id)
    })
}

//======================================================================================

//Multiple instance protection
const onlyInstance = app.requestSingleInstanceLock()
console.log(onlyInstance)

if (!onlyInstance)
{
    app.quit()
}
else
{
    setupApiHandlers()

    //App event listeners
    //=========================================================================================
    app.on('second-instance', (event, commandLine, workingDirectory, additionalData)=>
    {
        event.preventDefault()
    })

    app.on('window-all-closed', ()=>{app.quit()})

    app.whenReady().then(()=>
    {
        session.defaultSession.setCertificateVerifyProc((request, callback) => {
            callback(0); // 0 means accept all certificates
        });
        
        createWindow("index.html", {title: "Dista1"})
    })
}