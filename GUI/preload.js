const {contextBridge, ipcRenderer} = require("electron")

contextBridge.exposeInMainWorld("fileAPI", {
    writeFile: (filePath, data)=>{
        ipcRenderer.invoke('write-file', filePath, data);
    }
})

contextBridge.exposeInMainWorld("historyAPI", {
    loadHistory: ()=>{
        return ipcRenderer.invoke('load-history');
    },
    addToHistory: (entry)=>{
        return ipcRenderer.invoke('add-to-history', entry);
    },
    deleteFromHistory: (id)=>{
        return ipcRenderer.invoke('delete-from-history', id);
    }
})

