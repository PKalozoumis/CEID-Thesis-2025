//Variables
//====================================================================================================
let open_conversation = null
let convo_id_to_object = {} //may not be necessary

let dataToSend = null
let entry = null

let inputBox = document.querySelector("#query_input")
let queryTab = document.querySelector("div#query_tab")
let collectionTab = document.querySelector("div#collection_tab")
let queryTabContents = document.querySelector("div#query_tab_content")
let collectionTabContents = document.querySelector("div#collection_tab_content")

//Document pagination
let currentPage = 0
const pageSize = 10
let totalPages = null

//CLASSES
//====================================================================================================

//Data class that stores information for each of the history entries
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
                resetSearchLayout(true)

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
            if (window.socketAPI.connected())
            {
                store_to_db_and_kill_stream()
            }

            //Then, we swap conversation
            open_conversation = this.id
            document.querySelector("#initial_search_layout").classList.add("hidden")
            document.querySelector("#response_search_layout").classList.remove("hidden")
            document.querySelector("#query_header").textContent = `"${this.query}"`
            let responseField = document.querySelector("#response")
            
            //Populate the response field
            //--------------------------------------------------------------------------------
            responseField.textContent = ""; // clear existing content

            //Regex to match your stored citation format
            const regex = /<CITE: (\d+)_(\d+)-(\d+)>/g;

            let lastIndex = 0;
            let match;

            while ((match = regex.exec(this.response)) !== null) {
                // Text before the citation
                if (match.index > lastIndex) {
                    const textNode = document.createTextNode(this.response.slice(lastIndex, match.index));
                    responseField.appendChild(textNode);
                }

                //Citation span
                const citationNode = document.createElement("span");
                citationNode.classList.add("citation");
                citationNode.dataset.doc = match[1];
                citationNode.dataset.start = match[2];
                citationNode.dataset.end = match[3];
                citationNode.textContent = `[${match[1]}]`;
                citationNode.addEventListener("click", citationClickListener);
                responseField.appendChild(citationNode);

                lastIndex = regex.lastIndex;
            }

            //Any remaining text after the last citation
            if (lastIndex < this.response.length) {
                responseField.appendChild(document.createTextNode(this.response.slice(lastIndex)));
            }
        })

        return entry_div
    }
}

//Data class that stores information for each document in the collection list
class MyDocument
{
    static doc_id_to_object = {}

    constructor(id, title)
    {
        this.id = id
        this.title = title
        this.element = this.#createElement();
        MyDocument.doc_id_to_object[this.id] = this
    }

    #createElement()
    {
        const entry_div = document.createElement("div")
        const title_h2 = document.createElement("h2")
        const id_h2 = document.createElement("h2")

        entry_div.classList.add("document_entry")
        entry_div.id = this.id.toString()
        title_h2.textContent = this.title

        id_h2.classList.add("doc_id_field")
        id_h2.textContent = `ID: ${this.id}`

        entry_div.appendChild(id_h2)
        entry_div.appendChild(title_h2)

        entry_div.addEventListener("click", async (event)=>
        {
            await openDocument(this.id)
        })

        return entry_div
    }
}

//Message handlers
//====================================================================================================
function define_handlers()
{
    window.socketAPI.on("connect", ()=>
    {
        console.log("%cConnected to server successfully", "color: green")
        console.log(`Query: %c"${dataToSend['query']['text']}"\n`, "color: green")
        window.socketAPI.emit('init_query', dataToSend)
    })

    //-----

    window.socketAPI.on("disconnect", ()=>
    {
        console.log("Disconnecting from server...")
    })

    //-----

    window.socketAPI.on("message", (data)=>
    {
        console.log(`%c[MESSAGE]%c: ${data}`, "color: cyan", "")
    })

    //-----

    window.socketAPI.on("fragment", (data)=>
    {
        let responseField = document.querySelector("#response")

        if (!responseField.lastChild || responseField.lastChild.nodeType !== Node.TEXT_NODE)
            responseField.appendChild(document.createTextNode(data))
        else
            responseField.lastChild.textContent += data

        entry.response += data
    })

    //-----
    
    window.socketAPI.on("fragment_with_citation", (data)=>
    {
        //data: {
        //  fragment: ...
        //  cite: {'doc', 'start', 'end'} All int
        //}

        let responseField = document.querySelector("#response")
        let match = data.fragment.match(/(.*)<citation>(.*)/)

        //Text before the citation
        if (match[1].length > 0)
        {
            if (!responseField.lastChild || responseField.lastChild.nodeType !== Node.TEXT_NODE)
                responseField.appendChild(document.createTextNode(match[1]))
            else
                responseField.lastChild.textContent += match[1]
        }

        //Citation
        let citationNode = document.createElement("span")
        citationNode.classList.add('citation')
        citationNode.setAttribute('data-doc', data.citation.doc)
        citationNode.setAttribute('data-start', data.citation.start)
        citationNode.setAttribute('data-end', data.citation.end)
        citationNode.addEventListener('click', citationClickListener)
        citationNode.textContent = `[${data.citation.doc}]`
        responseField.appendChild(citationNode)

        //Text after the citation
        if (match[2].length > 0)
            responseField.appendChild(document.createTextNode(match[2]))

        entry.response += data.fragment.replace("<citation>", match[1] + `<CITE: ${data.citation.doc}_${data.citation.start}-${data.citation.end}>`)
    })

    //-----

    window.socketAPI.on("end", (data)=>
    {
        store_to_db_and_kill_stream()
        entry = null
        //Conversation remains open
    })
}

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

//====================================================================================================

function store_to_db_and_kill_stream(deleted = false)
{
    //The entry was not deleted
    //This means we should store it to the database
    if (!deleted)
    {
        //console.log("This will be stored")
        //console.log(convo_id_to_object[open_conversation])
        addToHistory(convo_id_to_object[open_conversation]).then(()=>{
            window.socketAPI.disconnect()
        })
    }
    else //Just close the stream
    {
        window.socketAPI.disconnect()
    }
}

//Document handling
//====================================================================================================

async function retrieveText(doc_id)
{
    let resp = await fetch(`https://localhost:9200/pubmed/_doc/${doc_id}?filter_path=_id,_source.article,_source.article_id`,
    {
        method: "GET",
        headers: {'Authorization': 'Basic ' + btoa("elastic" + ':' + "elastic"), 'Content-Type': 'application/json'}
    })
    let data = await resp.json()
    data = {id: data._id, title: data._source.article_id, txt: data._source.article}
    return data
}

async function openDocument(id, low = null, high = null)
{
    let data = await retrieveText(id)
    document.querySelector("#article_container").scrollTop = 0
    document.querySelector("#article_header").textContent = `${data.id} - ${data.title}`
    document.querySelector("#article").textContent = ""

    let span = null

    //Insert text and highlight
    if (low && high)
    {
        let txt1 = ""
        let txt2 = ""
        span = document.createElement('span')
        span.classList.add('highlight')
        span.textContent = ""
        let sents = data.txt.split('\n')

        for (let i = 0; i < sents.length; i++)
        {
            if (i < low)
                txt1 += sents[i]
            else if (i > high)
                txt2 += sents[i]
            else
            {
                span.textContent += sents[i]
            }
            
        }

        document.querySelector("#article").appendChild(document.createTextNode(txt1))
        document.querySelector("#article").appendChild(span)
        document.querySelector("#article").appendChild(document.createTextNode(txt2))
    }
    else
    {
        document.querySelector("#article").textContent = data.txt
    }

    document.querySelector("#collection_layout").classList.add("hidden")
    document.querySelector("#open_doc_layout").classList.remove("hidden")

    if (span)
        span.scrollIntoView({behavior: "auto", block: "center" });
}

//Collection tab pagination
//===================================================================================================
function renderPagination() {
    const container = document.querySelector("#pagination");
    container.innerHTML = "";

    // Prev
    const prevBtn = document.createElement("button");
    prevBtn.classList.add("paginationButton")
    prevBtn.textContent = "<";
    prevBtn.disabled = currentPage === 0;
    prevBtn.onclick = () => { currentPage--; loadPage(); };
    container.appendChild(prevBtn);

    // Pages (only show a window around current page)
    const windowSize = 5;
    let start = Math.max(0, currentPage - Math.floor(windowSize/2));
    let end = Math.min(totalPages, start + windowSize);
    if (end - start < windowSize) start = Math.max(0, end - windowSize);

    for (let i = start; i < end; i++) {
        const btn = document.createElement("button");
        btn.classList.add("paginationButton")
        btn.textContent = i + 1;
        if (i === currentPage) btn.classList.add("active");
        btn.onclick = () => { currentPage = i; loadPage(); };
        container.appendChild(btn);
    }

    // Next
    const nextBtn = document.createElement("button");
    nextBtn.classList.add("paginationButton")
    nextBtn.textContent = ">";
    nextBtn.disabled = currentPage >= totalPages - 1;
    nextBtn.onclick = () => { currentPage++; loadPage(); };
    container.appendChild(nextBtn);
}

//Load the next set of documents in the collection list
async function loadPage()
{
    let resp = await fetch("https://localhost:9200/pubmed/_search?filter_path=hits.total.value,hits.hits._id,hits.hits._source.article_id",
    {
        method: "POST",
        headers: {'Authorization': 'Basic ' + btoa("elastic" + ':' + "elastic"), 'Content-Type': 'application/json'},
        body: JSON.stringify({"from": currentPage*pageSize, "size": pageSize})
    })
    let data = await resp.json()

    totalPages = Math.ceil(data.hits.total.value / pageSize);
    
    //Extract only id and title from the results
    data = data.hits.hits.map((item)=>({
        id: item._id,
        title: item._source.article_id
    }))

    //Create Document objects
    data = data.map(item => new MyDocument(item.id, item.title))
    
    //Place them in the layout
    const frag = document.createDocumentFragment()
    for (let obj of data)
        frag.appendChild(obj.element)
    container = document.querySelector("#document_list")
    container.innerHTML = "";
    container.appendChild(frag);

    renderPagination()
}

//Tab selection
//====================================================================================================
queryTab.addEventListener("click", (event)=>
{
    queryTab.classList.add("selected");
    queryTabContents.classList.remove("hidden")
    collectionTab.classList.remove("selected");
    collectionTabContents.classList.add("hidden");
})

//---------------------------------------------------------------------

async function switchToCollectionTab(event)
{
    collectionTab.classList.add("selected");
    collectionTabContents.classList.remove("hidden");
    queryTab.classList.remove("selected");
    queryTabContents.classList.add("hidden");

    loadPage()
}

collectionTab.addEventListener("click", switchToCollectionTab)

//Search tab
//====================================================================================================
async function citationClickListener(event)
{
    await switchToCollectionTab(event)
    await openDocument(event.target.dataset.doc, event.target.dataset.start, event.target.dataset.end)
}

//Leave open conversation. Return to layout with the query input box.
function resetSearchLayout(deleted = false)
{
    document.querySelector("#initial_search_layout").classList.remove("hidden")
    document.querySelector("#response_search_layout").classList.add("hidden")
    inputBox.innerText = ""

    //We are in the middle of an ongoing conversation
    if (window.socketAPI.connected())
    {
        store_to_db_and_kill_stream(deleted)
    }

    open_conversation = null
}

inputBox.addEventListener("keydown", async (event)=>
{   
    if (event.key === 'Enter')
    {
        event.preventDefault()
        const text = inputBox.innerText.trim()
        if (text)
        {
            document.querySelector("#initial_search_layout").classList.add("hidden")
            document.querySelector("#response_search_layout").classList.remove("hidden")
            document.querySelector("#query_header").textContent = `"${text}"`

            dataToSend = {
                query: {
                    id: -1,
                    text: text,
                    source: ["article", "summary"],
                    text_path: "article"
                },
                args: {
                    print: false,
                    experiment: 'default' 
                }
            }

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

            entry = new HistoryEntry(text, formatted)
            const history_entries = document.querySelector("#history_entries")
            history_entries.insertBefore(entry.element, history_entries.firstChild)
            
            //Set the new conversation as selected
            open_conversation = entry.id

            //Connect
            window.socketAPI.connect()
            define_handlers()
        }
    }
})

document.querySelector("#new_query").addEventListener("click", (event)=>{
    if (open_conversation !== null)
        resetSearchLayout()
})

//Collection tab
//====================================================================================================
//Leave open document. Return to layout with the list of documents
function resetCollectionLayout(deleted = false)
{
    document.querySelector("#article_container").scrollTop = 0
    document.querySelector("#collection_layout").classList.remove("hidden")
    document.querySelector("#open_doc_layout").classList.add("hidden")
    inputBox.innerText = ""
}

document.querySelector("#new_document").addEventListener("click", (event)=>{
    resetCollectionLayout()
})

//Query history
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
    //console.log(`Next history entry: ${HistoryEntry.current_id}`);
}

async function addToHistory(data)
{
    HistoryEntry.current_id = await window.historyAPI.addToHistory(data);
}

//Main
//====================================================================================================
loadHistory()