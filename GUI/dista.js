//Variables
//====================================================================================================
let open_conversation = null
let convo_id_to_object = {} //may not be necessary

let stream = null

let inputBox = document.querySelector("#query_input")

let queryTab = document.querySelector("div#query_tab")
let collectionTab = document.querySelector("div#collection_tab")
let queryTabContents = document.querySelector("div#query_tab_content")
let collectionTabContents = document.querySelector("div#collection_tab_content")

function store_to_db_and_kill_stream(deleted = false)
{
    //The entry was not deleted
    //This means we should store it to the database
    if (!deleted)
    {
        console.log("This will be stored")
        console.log(convo_id_to_object[open_conversation])
        addToHistory(convo_id_to_object[open_conversation]).then(()=>{
            stream.close()
            stream = null
        })
    }
    else //Just close the stream
    {
        stream.close()
        stream = null
    }
}

function reset_search_layout(deleted = false)
{
    document.querySelector("#initial_search_layout").classList.remove("hidden")
    document.querySelector("#response_search_layout").classList.add("hidden")
    inputBox.innerText = ""

    //We are in the middle of an ongoing conversation
    if (stream !== null)
    {
        store_to_db_and_kill_stream(deleted)
    }

    open_conversation = null
}

class HistoryEntry
{
    static current_id = 0;

    constructor(query, time, response="", id=null)
    {
        this.id = id === null ? HistoryEntry.current_id++ : id;
        this.query = query;
        this.time = time;
        this.response = response;
        this.element = this.#createElement();

        convo_id_to_object[this.id] = this
    }

    #createElement()
    {
        const entry_div = document.createElement("div")
        const title_h2 = document.createElement("h2")
        const lower_div = document.createElement("div")
        const date_p = document.createElement("p")
        const delete_span = document.createElement("span")

        entry_div.classList.add("sidebar_entry")
        entry_div.id = this.id.toString()
        title_h2.textContent = this.query
        date_p.textContent = this.time
        delete_span.classList.add("material-symbols-outlined")
        delete_span.textContent = "delete"

        delete_span.addEventListener("click", (event)=>
        {
            //Check if you are about to delete the current conversation
            if (this.id == open_conversation)
                reset_search_layout(true)

            //The this here works because it's an arrow function. It perserves the reference to the object
            this.element.remove()

            //Remove from the database
            //May not delete anything if the current convo is not done yet
            window.historyAPI.deleteFromHistory(this.id)

            //Prevent the whole entry from being clicked
            event.stopPropagation()
        })

        lower_div.appendChild(date_p)
        lower_div.appendChild(delete_span)
        entry_div.appendChild(title_h2)
        entry_div.appendChild(lower_div)

        entry_div.addEventListener("click", (event)=>
        {
            //Check if you are about to click on the current conversation
            //If so do nothing
            if (this.id == open_conversation)
                return
            
            //Otherwise, we need to check if a stream is currently running
            //We need to stop that stream
            if (stream !== null)
            {
                store_to_db_and_kill_stream()
            }

            //Then, we swap conversation
            open_conversation = this.id
            document.querySelector("#initial_search_layout").classList.add("hidden")
            document.querySelector("#response_search_layout").classList.remove("hidden")
            document.querySelector("#query_header").textContent = `"${this.query}"`
            let responseField = document.querySelector("#response")
            responseField.textContent = this.response
        })

        return entry_div
    }
}


//Tab selection
//====================================================================================================
queryTab.addEventListener("click", (event)=>
{
    queryTab.classList.add("selected");
    queryTabContents.classList.remove("hidden-contents")
    collectionTab.classList.remove("selected");
    collectionTabContents.classList.add("hidden-contents");
})

//---------------------------------------------------------------------

collectionTab.addEventListener("click", async (event)=>
{
    collectionTab.classList.add("selected");
    collectionTabContents.classList.remove("hidden-contents");
    queryTab.classList.remove("selected");
    queryTabContents.classList.add("hidden-contents");

    let resp = await fetch("https://localhost:9200/pubmed/_search?filter_path=hits.hits._id",
    {
        method: "POST",
        headers: {'Authorization': 'Basic ' + btoa("elastic" + ':' + "elastic"), 'Content-Type': 'application/json'},
        body: JSON.stringify({"from": 0, "size": 10})
    })
    let data = await resp.json()
    console.log(data)
})

//Remove formatting from pasted text
//====================================================================================================
inputBox.addEventListener("paste", (event)=>{
    event.preventDefault()
    const text = event.clipboardData.getData("text/plain")
    document.execCommand('insertText', false, text);
})

document.addEventListener("copy", (event)=>
{
    event.preventDefault()
    const copiedText = window.getSelection().toString();
    event.clipboardData.setData('text/plain', copiedText);
})

//Send query
//====================================================================================================
inputBox.addEventListener("keydown", async (event)=>
{   
    if (event.key === 'Enter')
    {
        event.preventDefault()
        const text = inputBox.innerText.trim()
        if (text)
        {
            console.log(text)
            document.querySelector("#initial_search_layout").classList.add("hidden")
            document.querySelector("#response_search_layout").classList.remove("hidden")
            document.querySelector("#query_header").textContent = `"${text}"`

            let params = new URLSearchParams({'q': text})

            let responseField = document.querySelector("#response")
            responseField.textContent = ""

            //Add history entry
            const now = new Date();

            const h = String(now.getHours()).padStart(2, '0');
            const m = String(now.getMinutes()).padStart(2, '0');
            const d = String(now.getDate()).padStart(2, '0');
            const mo = String(now.getMonth() + 1).padStart(2, '0');
            const y = now.getFullYear();

            const formatted = `${h}:${m} ${d}/${mo}/${y}`;

            const entry = new HistoryEntry(text, formatted)
            const history_entries = document.querySelector("#history_entries")
            history_entries.insertBefore(entry.element, history_entries.firstChild)
            
            //Select the current conversation
            open_conversation = entry.id
            console.log(open_conversation)

            stream = new EventSource(`http://localhost:4625/query?${params}`);     

            //For every message you receive, do this
            stream.onmessage = (event)=>{
                let data = JSON.parse(event.data)
                switch (data.type)
                {
                    case "fragment":
                    {
                        responseField.textContent += data.contents
                        entry.response += data.contents
                    }
                    break;
                    case "end":
                    {
                        store_to_db_and_kill_stream()
                        //Conversation remains open
                    }
                    break;
                }
            }
        }
    }
})


//New query
//====================================================================================================
document.querySelector("#new_query").addEventListener("click", (event)=>{
    if (open_conversation !== null)
        reset_search_layout()
})

//Load history
//====================================================================================================
async function loadHistory()
{
    let data = await window.historyAPI.loadHistory()
    data = data.map(item => new HistoryEntry(item.query, item.time, item.response, item.id))
    HistoryEntry.current_id = data[0].id + 1
    
    const frag = document.createDocumentFragment()

    for (let obj of data)
        frag.appendChild(obj.element)

    document.querySelector("#history_entries").appendChild(frag);

    //window.fileAPI.writeFile("temp/dista.json", JSON.stringify({'dista': 1}))

    console.log(`Next history entry: ${HistoryEntry.current_id}`);
}

//Add to history
//====================================================================================================
async function addToHistory(data)
{
    HistoryEntry.current_id = await window.historyAPI.addToHistory(data);
}

//Main
//====================================================================================================
loadHistory()