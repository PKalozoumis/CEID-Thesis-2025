const {contextBridge, ipcRenderer} = require("electron")

const io = require('socket.io-client');

//===========================================================================================

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


let sio

contextBridge.exposeInMainWorld("socketAPI", {
    connect: () => {
        sio = io(`http://localhost:1225/query`);

        //Can also do this, to connect manually when needed:
        //const socket = io("http://localhost:3000", { autoConnect: false });
        //socket.connect();
    },

    on: (event, callback) => {
        if (sio) sio.on(event, callback);
    },

    emit: (event, data) => {
        //Automatically sends to the correct namespace
        if (sio) sio.emit(event, data);
    },

    connected: () => {
        return sio ? sio.connected : false;
    },

    disconnect: () => {
        sio.disconnect()
    }
});
