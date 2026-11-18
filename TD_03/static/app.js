// ============================================================
// RAG VISUAL LAB — APPLICATION
// PASS 4: ANIMATED VECTOR SEARCH
// ============================================================

const $ = id => document.getElementById(id);

const stages = [...document.querySelectorAll("#stages div")];

const sleep = ms =>
    new Promise(resolve => setTimeout(resolve, ms));

let currentChunks = [];
let currentResults = [];
let selectedResultIndex = null;
let busy = false;

// PCA and vector-search state.
let documentPoints = [];

const rankingComparison = {
    data: null,
    mode: "pca_euclidean",
    view: "both",
    selected: null
};

const COMPARISON_COLORS = {
    original: "#60a5fa",
    projected: "#a78bfa",
    shared: "#4adeb6"
};

const vectorSearch = {
    data: null,
    order: [],
    position: 0,
    examined: [],
    playing: false,
    timer: null,
    generation: 0,
    currentChunk: null,
    maxAnimatedSteps: 80
};

const SEARCH_COLORS = {
    document: "#65d9c4",
    query: "#fbbf77",
    examined: "#60a5fa",
    current: "#fbbf77",
    topK: "#4adeb6",
    connection: "#a78bfa"
};


// ============================================================
// API HELPERS
// ============================================================

async function api(path, body, isFile = false) {

    const response = await fetch("/api/" + path, {
        method: "POST",

        headers: isFile
            ? {}
            : { "Content-Type": "application/json" },

        body: isFile
            ? body
            : JSON.stringify(body)
    });

    let data;

    try {
        data = await response.json();
    } catch {
        throw Error(`Server error (${response.status}).`);
    }

    if (!response.ok) {
        throw Error(data.error || "Request failed.");
    }

    return data;
}


function setStatus(message) {
    $("status").textContent = message;
}


function stage(n) {

    stages.forEach((element, index) => {

        element.classList.toggle("done", index < n);
        element.classList.toggle("active", index === n);

    });
}


function add(parent, className, text = "") {

    const element = document.createElement("div");

    element.className = className;
    element.textContent = text;

    parent.append(element);

    return element;
}


function createButton(parent, className, text, handler) {

    const button = document.createElement("button");

    button.type = "button";
    button.className = className;
    button.textContent = text;

    button.addEventListener("click", handler);

    parent.append(button);

    return button;
}


function run(fn) {

    return async () => {

        try {
            await fn();
        } catch (error) {
            console.error(error);
            setStatus(error.message);
        }

    };
}


// ============================================================
// DOCUMENT EXPLORER TABS
// ============================================================

const explorerTabs = [
    ...document.querySelectorAll(".explorer-tab")
];

const explorerPanels = [
    ...document.querySelectorAll(".explorer-panel")
];


function showTab(name) {

    explorerTabs.forEach(button => {

        const selected = button.dataset.tab === name;

        button.classList.toggle("active", selected);

        button.setAttribute(
            "aria-selected",
            String(selected)
        );

        button.tabIndex = selected ? 0 : -1;
    });

    explorerPanels.forEach(panel => {

        const selected = panel.id === `tab-${name}`;

        panel.hidden = !selected;
        panel.classList.toggle("active", selected);
    });

    if (
        name === "pca" &&
        window.Plotly &&
        $("plot").data
    ) {

        requestAnimationFrame(() => {
            Plotly.Plots.resize($("plot"));
        });
    }
}


explorerTabs.forEach((button, index) => {

    button.addEventListener("click", () => {
        showTab(button.dataset.tab);
    });

    button.addEventListener("keydown", event => {

        const keys = [
            "ArrowLeft",
            "ArrowRight",
            "Home",
            "End"
        ];

        if (!keys.includes(event.key)) {
            return;
        }

        event.preventDefault();

        let next;

        if (event.key === "Home") {

            next = 0;

        } else if (event.key === "End") {

            next = explorerTabs.length - 1;

        } else {

            const direction =
                event.key === "ArrowRight" ? 1 : -1;

            next = (
                index +
                direction +
                explorerTabs.length
            ) % explorerTabs.length;
        }

        explorerTabs[next].focus();

        showTab(explorerTabs[next].dataset.tab);
    });
});


// ============================================================
// TEXT HIGHLIGHTING
// ============================================================

function appendHighlightedText(
    parent,
    text,
    query,
    overlapLength = 0
) {

    text = String(text || "");

    const needle = query.toLocaleLowerCase();
    const lower = text.toLocaleLowerCase();

    let position = 0;

    while (position < text.length) {

        const match = needle
            ? lower.indexOf(needle, position)
            : -1;

        const nextMatch = match < 0
            ? text.length
            : match;

        const boundary = position < overlapLength
            ? Math.min(nextMatch, overlapLength)
            : nextMatch;

        if (boundary > position) {

            const fragment = text.slice(
                position,
                boundary
            );

            if (position < overlapLength) {

                const mark = document.createElement("mark");

                mark.textContent = fragment;
                parent.append(mark);

            } else {

                parent.append(
                    document.createTextNode(fragment)
                );
            }

            position = boundary;
            continue;
        }

        if (match === position && needle) {

            const end = Math.min(
                text.length,
                position + query.length
            );

            const mark = document.createElement("mark");

            mark.className = "search-highlight";

            mark.textContent = text.slice(
                position,
                end
            );

            parent.append(mark);

            position = end;

        } else if (position < overlapLength) {

            const mark = document.createElement("mark");

            mark.textContent = text.slice(
                position,
                overlapLength
            );

            parent.append(mark);

            position = overlapLength;

        } else {

            parent.append(
                document.createTextNode(
                    text.slice(position)
                )
            );

            break;
        }
    }
}


// ============================================================
// CHUNK EXPLORER
// ============================================================

function renderChunks() {

    const query = $("chunk-search").value.trim();
    const normalized = query.toLocaleLowerCase();

    const matches = currentChunks.filter(chunk =>
        chunk.text.toLocaleLowerCase().includes(normalized)
    );

    $("chunks").replaceChildren();

    $("chunk-search-count").textContent =
        currentChunks.length
            ? `Showing ${matches.length} of ${currentChunks.length}`
            : "";

    if (!matches.length && currentChunks.length) {

        add(
            $("chunks"),
            "hint",
            "No chunks match your search."
        );
    }

    matches.forEach(chunk => {

        const previous = currentChunks[chunk.index - 1];

        const shared = previous
            ? Math.max(0, previous.end - chunk.start)
            : 0;

        const card = document.createElement("button");

        card.type = "button";
        card.className = "chunkcard";

        add(
            card,
            "chunkheader",
            `Chunk ${chunk.index + 1} · ` +
            `characters ${chunk.start}–${chunk.end} · ` +
            `${chunk.text.length} chars`
        );

        const body = add(card, "chunkbody");

        appendHighlightedText(
            body,
            chunk.text,
            query,
            shared
        );

        card.onclick = () => {

            $("chunks")
                .querySelectorAll(".chunkcard")
                .forEach(element => {
                    element.classList.remove("selected");
                });

            card.classList.add("selected");

            $("chunkdetail").hidden = false;
            $("chunkdetail").replaceChildren();

            add(
                $("chunkdetail"),
                "detailtitle",
                `Chunk ${chunk.index + 1} · ` +
                `${shared} overlapping characters ` +
                `from previous chunk`
            );

            const content = add(
                $("chunkdetail"),
                "detailtext"
            );

            appendHighlightedText(
                content,
                chunk.text,
                query,
                shared
            );
        };

        $("chunks").append(card);
    });

    $("chunkdetail").replaceChildren();
    $("chunkdetail").hidden = true;
}


$("chunk-search").addEventListener(
    "input",
    renderChunks
);


function resetChunkSearch() {

    currentChunks = [];

    $("chunk-search").value = "";
    $("chunk-search").disabled = true;

    $("chunk-search-count").textContent = "";

    $("chunks").replaceChildren();

    $("chunkdetail").replaceChildren();
    $("chunkdetail").hidden = true;
}


// ============================================================
// EMBEDDING INFORMATION
// ============================================================

function resetEmbeddingInfo() {

    $("embedding-status").textContent =
        "Generate embeddings to inspect their " +
        "numerical representation.";

    $("embedding-stats").replaceChildren();
}


function updateEmbeddingInfo(data) {

    $("embedding-status").textContent =
        "Document chunks have been converted " +
        "into numerical embeddings.";

    $("embedding-dimension").textContent =
        String(data.dimension);

    $("embedding-stats").replaceChildren();

    const statistics = [
        ["Vectors", data.points.length],
        ["Dimensions", data.dimension]
    ];

    statistics.forEach(([label, value]) => {

        const box = add(
            $("embedding-stats"),
            "stat",
            String(value)
        );

        add(box, "statlabel", label);
    });
}


// ============================================================
// PCA MANAGEMENT
// ============================================================

function resetPlot() {

    resetVectorSearch();
    resetRankingComparison();

    documentPoints = [];

    if (window.Plotly && $("plot").data) {
        Plotly.purge($("plot"));
    }

    $("plot").replaceChildren();
}


// ============================================================
// PCA PLOT TRACES
// ============================================================

// Trace indices are kept stable:
//
// 0 — Document chunks
// 1 — Examined chunks
// 2 — Final top-k chunks
// 3 — Current comparison line
// 4 — Current comparison point
// 5 — Query embedding
//
// Keeping trace indices stable makes animation updates
// straightforward.

function createPCATraces() {

    const x = documentPoints.map(point => point[0]);
    const y = documentPoints.map(point => point[1]);

    return [

        // ----------------------------------------------------
        // TRACE 0: ALL DOCUMENT CHUNKS
        // ----------------------------------------------------

        {
            x,
            y,

            type: "scatter",
            mode: "markers",

            name: "Document chunks",

            text: documentPoints.map(
                (_, index) => `Chunk ${index + 1}`
            ),

            customdata: documentPoints.map(
                (_, index) => index
            ),

            hovertemplate:
                "%{text}<br>" +
                "PC1: %{x:.3f}<br>" +
                "PC2: %{y:.3f}" +
                "<extra></extra>",

            marker: {
                size: 9,
                color: SEARCH_COLORS.document,
                opacity: 0.75,

                line: {
                    width: 1,
                    color: "#b9fff0"
                }
            }
        },

        // ----------------------------------------------------
        // TRACE 1: EXAMINED CHUNKS
        // ----------------------------------------------------

        {
            x: [],
            y: [],

            type: "scatter",
            mode: "markers",

            name: "Examined",

            hoverinfo: "skip",

            marker: {
                size: 12,
                color: SEARCH_COLORS.examined,
                opacity: 0.9,

                line: {
                    width: 1,
                    color: "#dbeafe"
                }
            }
        },

        // ----------------------------------------------------
        // TRACE 2: FINAL TOP-K
        // ----------------------------------------------------

        {
            x: [],
            y: [],

            type: "scatter",
            mode: "markers",

            name: "Top-k",

            text: [],
            customdata: [],

            hovertemplate:
                "%{text}<br>" +
                "PC1: %{x:.3f}<br>" +
                "PC2: %{y:.3f}" +
                "<extra></extra>",

            marker: {
                size: 16,
                color: SEARCH_COLORS.topK,
                opacity: 1,

                symbol: "circle",

                line: {
                    width: 2,
                    color: "#eafff8"
                }
            }
        },

        // ----------------------------------------------------
        // TRACE 3: QUERY → CURRENT CHUNK
        // ----------------------------------------------------

        {
            x: [],
            y: [],

            type: "scatter",
            mode: "lines",

            name: "Current comparison",

            hoverinfo: "skip",

            line: {
                color: SEARCH_COLORS.connection,
                width: 2,
                dash: "dash"
            }
        },

        // ----------------------------------------------------
        // TRACE 4: CURRENT CHUNK
        // ----------------------------------------------------

        {
            x: [],
            y: [],

            type: "scatter",
            mode: "markers",

            name: "Current chunk",

            text: [],

            hovertemplate:
                "%{text}<extra></extra>",

            marker: {
                size: 18,
                color: SEARCH_COLORS.current,

                symbol: "circle-open",

                line: {
                    width: 3,
                    color: SEARCH_COLORS.current
                }
            }
        },

        // ----------------------------------------------------
        // TRACE 5: QUERY EMBEDDING
        // ----------------------------------------------------

        {
            x: [],
            y: [],

            type: "scatter",
            mode: "markers+text",

            name: "Query",

            text: ["Query"],
            textposition: "top center",

            hovertemplate:
                "Query embedding<br>" +
                "PC1: %{x:.3f}<br>" +
                "PC2: %{y:.3f}" +
                "<extra></extra>",

            marker: {
                size: 17,
                color: SEARCH_COLORS.query,

                symbol: "diamond",

                line: {
                    width: 2,
                    color: "#fff2d5"
                }
            },

            textfont: {
                color: SEARCH_COLORS.query,
                size: 11
            }
        }
    ];
}

function createComparisonTraces() {

    return [
        {
            name: "Actual retrieval only",
            color: COMPARISON_COLORS.original
        },
        {
            name: "PCA neighbors only",
            color: COMPARISON_COLORS.projected
        },
        {
            name: "Shared neighbors",
            color: COMPARISON_COLORS.shared
        }
    ].map(item => ({

        x: [],
        y: [],

        customdata: [],
        text: [],

        type: "scatter",
        mode: "markers",

        name: item.name,

        hovertemplate:
            "%{text}<extra></extra>",

        marker: {
            size: 19,
            symbol: "circle-open",

            color: item.color,

            line: {
                color: item.color,
                width: 4
            }
        }
    }));
}


// ============================================================
// RENDER PCA
// ============================================================

async function renderPCA(data) {

    resetVectorSearch();
    resetRankingComparison();

    documentPoints = data.points.map(point => [
        Number(point[0]),
        Number(point[1])
    ]);

    showTab("pca");

    await new Promise(resolve => {
        requestAnimationFrame(resolve);
    });

    if (!window.Plotly) {

        $("plot").textContent =
            "Plotly unavailable. Check internet access.";

        return;
    }

    await Plotly.newPlot(
        "plot",

        createPCATraces(),

        {
            paper_bgcolor: "#151f31",
            plot_bgcolor: "#151f31",

            font: {
                color: "#e8edf6"
            },

            margin: {
                l: 40,
                r: 15,
                t: 15,
                b: 40
            },

            xaxis: {
                title: "PC1",
                zeroline: false,
                gridcolor: "#293b53"
            },

            yaxis: {
                title: "PC2",
                zeroline: false,
                gridcolor: "#293b53"
            },

            hovermode: "closest",

            showlegend: false,

            uirevision: "rag-pca"
        },

        {
            responsive: true,
            displaylogo: false,

            modeBarButtonsToRemove: [
                "lasso2d",
                "select2d"
            ]
        }
    );

    await Plotly.addTraces(
        $("plot"),
        createComparisonTraces()
    );

    Plotly.Plots.resize($("plot"));

    // Clicking a document point opens its original text.
    //
    // Plotly click events contain the trace index and
    // point index, which let us recover the chunk index.

    const plot = $("plot");

    if (typeof plot.removeAllListeners === "function") {
        plot.removeAllListeners("plotly_click");
    }

    plot.on("plotly_click", event => {

        const point = event.points?.[0];

        if (!point) {
            return;
        }

        const traceIndex = point.curveNumber;

        let chunkIndex = null;

        if (traceIndex === 0) {

            chunkIndex = point.pointIndex;

        } else if (traceIndex === 2) {

            chunkIndex = point.data.customdata?.[
                point.pointIndex
            ];

        } else if (traceIndex === 4) {

            chunkIndex = vectorSearch.currentChunk;

        } else if (traceIndex >= 6 && traceIndex <= 8) {

            chunkIndex = point.data.customdata?.[
                point.pointIndex
            ];
        }

        if (chunkIndex !== null && chunkIndex !== undefined) {
            inspectChunkFromPlot(Number(chunkIndex));
        }
    });
}


// ============================================================
// INSPECT CHUNK FROM PCA
// ============================================================

function inspectChunkFromPlot(chunkIndex) {

    const chunk = currentChunks.find(
        item => item.index === chunkIndex
    );

    if (!chunk) {
        return;
    }

    showTab("chunks");

    $("chunk-search").value = "";

    renderChunks();

    const cards = [
        ...$("chunks").querySelectorAll(".chunkcard")
    ];

    const card = cards[chunkIndex];

    if (card) {

        card.click();

        card.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }
}


// ============================================================
// PASS 4 — VECTOR SEARCH ANIMATION
// ============================================================

// The backend performs the real search.
//
// JavaScript receives:
//
// - query_point: query coordinates in the document PCA
// - similarities: all chunks with real cosine scores
// - results: final top-k chunks
// - dimension: original embedding dimension
//
// The animation replays those comparisons.
//
// IMPORTANT:
// The 2D plot is illustrative.
// Similarity scores come from the original 384D vectors.


// ============================================================
// ANIMATION UTILITIES
// ============================================================

function searchSpeedDelay() {

    const speed = $("search-speed").value;

    if (speed === "slow") {
        return 900;
    }

    if (speed === "fast") {
        return 140;
    }

    return 400;
}


function clearSearchTimer() {

    if (vectorSearch.timer !== null) {

        clearTimeout(vectorSearch.timer);

        vectorSearch.timer = null;
    }
}


function pauseVectorSearch() {

    vectorSearch.playing = false;

    clearSearchTimer();

    vectorSearch.generation += 1;

    updateSearchControls();

    if (
        vectorSearch.data &&
        vectorSearch.position < vectorSearch.order.length
    ) {

        $("vector-search-state").textContent = "Paused";
    }
}


// ============================================================
// RESET ANIMATION
// ============================================================

function resetVectorSearch() {

    pauseVectorSearch();

    vectorSearch.data = null;
    vectorSearch.order = [];
    vectorSearch.position = 0;
    vectorSearch.examined = [];
    vectorSearch.currentChunk = null;

    $("vector-search-state").textContent =
        "Waiting for a question";

    $("vector-search-query-text").textContent =
        "Ask a question to see its query embedding here.";

    $("search-progress-text").textContent =
        "No search to replay yet.";

    $("search-progress-count").textContent = "0 / 0";

    $("search-progress").max = 1;
    $("search-progress").value = 0;

    $("search-comparison").hidden = true;

    $("search-current-chunk").textContent = "Chunk —";

    $("search-current-score").textContent =
        "Cosine similarity: —";

    $("search-ranking").replaceChildren();

    $("search-ranking-caption").textContent =
        "Highest-scoring chunks examined so far";

    updateSearchControls();
}


// ============================================================
// ANIMATION CONTROLS
// ============================================================

function updateSearchControls() {

    const available = Boolean(vectorSearch.data);

    const finished =
        available &&
        vectorSearch.position >= vectorSearch.order.length;

    $("search-play").disabled =
        !available ||
        vectorSearch.playing ||
        finished;

    $("search-pause").disabled =
        !available ||
        !vectorSearch.playing;

    $("search-next").disabled =
        !available ||
        vectorSearch.playing ||
        finished;

    $("search-replay").disabled = !available;

    $("search-speed").disabled = !available;
}


// ============================================================
// PREPARE SEARCH ANIMATION
// ============================================================

function prepareVectorSearch(data) {

    resetVectorSearch();

    if (
        !Array.isArray(data.similarities) ||
        !Array.isArray(data.query_point) ||
        !data.similarities.length
    ) {

        console.warn(
            "No vector-search visualization data returned."
        );

        return;
    }

    if (
        !Array.isArray(documentPoints) ||
        !documentPoints.length
    ) {

        console.warn(
            "Document PCA coordinates are unavailable."
        );

        return;
    }

    const similarities = data.similarities.map(item => ({
        index: Number(item.index),
        score: Number(item.score),
        rank: Number(item.rank)
    }));

    const valid = similarities.every(item =>
        Number.isInteger(item.index) &&
        item.index >= 0 &&
        item.index < documentPoints.length &&
        Number.isFinite(item.score)
    );

    if (!valid) {

        console.warn(
            "Invalid similarity data returned by backend."
        );

        return;
    }

    // The backend returns similarities sorted by score.
    //
    // For the educational animation, we examine chunks
    // in their original document order instead.
    //
    // This avoids revealing the final ranking in advance.

    const byDocumentOrder = [...similarities].sort(
        (a, b) => a.index - b.index
    );

    const queryPoint = data.query_point.map(Number);

    if (
        queryPoint.length !== 2 ||
        !queryPoint.every(Number.isFinite)
    ) {

        console.warn("Invalid query PCA coordinates.");

        return;
    }

    vectorSearch.data = {
        queryPoint,

        similarities,

        results: data.results || [],

        searchQuery: data.search_query || "",

        dimension: Number(data.dimension) || 384,

        topK: (data.results || []).length
    };

    // For large documents, replay a representative subset.
    //
    // The real backend search still evaluated ALL chunks.
    //
    // We always include the final top-k results in the
    // animated subset, then fill remaining slots with
    // evenly distributed document chunks.

    vectorSearch.order = selectAnimationSteps(
        byDocumentOrder,
        vectorSearch.data.results,
        vectorSearch.maxAnimatedSteps
    );

    vectorSearch.position = 0;
    vectorSearch.examined = [];
    vectorSearch.currentChunk = null;

    $("vector-search-query-text").textContent =
        vectorSearch.data.searchQuery;

    $("vector-search-state").textContent = "Ready";

    $("search-progress-text").textContent =
        "Ready to replay vector search.";

    $("search-progress").max =
        Math.max(1, vectorSearch.order.length);

    $("search-progress").value = 0;

    $("search-progress-count").textContent =
        `0 / ${vectorSearch.order.length}`;

    $("search-ranking").replaceChildren();

    if (vectorSearch.order.length < similarities.length) {

        $("search-ranking-caption").textContent =
            `Showing ${vectorSearch.order.length} of ` +
            `${similarities.length} comparisons`;

    } else {

        $("search-ranking-caption").textContent =
            "Highest-scoring chunks examined so far";
    }

    updateSearchControls();

    // Display the query immediately.
    //
    // Students can then press Play whenever they want.

    updateVectorPlot();

    // Pass 5: Compare original and PCA rankings.
    prepareRankingComparison(data);
}


// ============================================================
// SELECT REPRESENTATIVE ANIMATION STEPS
// ============================================================

function selectAnimationSteps(
    allComparisons,
    topResults,
    maxSteps
) {

    if (allComparisons.length <= maxSteps) {
        return [...allComparisons];
    }

    const selected = new Set();

    // Always include every top-k chunk.
    topResults.forEach(result => {
        selected.add(Number(result.index));
    });

    // Distribute the remaining animated examples
    // across the document.

    const availableSlots = Math.max(
        0,
        maxSteps - selected.size
    );

    if (availableSlots > 0) {

        for (let i = 0; i < availableSlots; i++) {

            const position = availableSlots === 1
                ? 0
                : Math.round(
                    i *
                    (allComparisons.length - 1) /
                    (availableSlots - 1)
                );

            selected.add(
                allComparisons[position].index
            );
        }
    }

    // If some evenly spaced choices coincided with top-k,
    // fill remaining slots in document order.

    for (const comparison of allComparisons) {

        if (selected.size >= maxSteps) {
            break;
        }

        selected.add(comparison.index);
    }

    return allComparisons.filter(
        item => selected.has(item.index)
    );
}


// ============================================================
// ANIMATION STEP
// ============================================================

function advanceVectorSearch() {

    if (!vectorSearch.data) {
        return;
    }

    if (vectorSearch.position >= vectorSearch.order.length) {
        finishVectorSearch();
        return;
    }

    const comparison =
        vectorSearch.order[vectorSearch.position];

    vectorSearch.position += 1;

    vectorSearch.examined.push(comparison);

    vectorSearch.currentChunk = comparison.index;

    // Update current comparison.

    $("search-comparison").hidden = false;

    $("search-current-chunk").textContent =
        `Chunk ${comparison.index + 1}`;

    $("search-current-score").textContent =
        `Cosine similarity: ${comparison.score.toFixed(4)}`;

    // Update progress.

    const examined = vectorSearch.position;
    const total = vectorSearch.order.length;

    $("search-progress").value = examined;

    $("search-progress-count").textContent =
        `${examined} / ${total}`;

    $("search-progress-text").textContent =
        `Examining chunk ${comparison.index + 1} ` +
        `(${examined} of ${total})`;

    $("vector-search-state").textContent = "Searching";

    // Update ranking and PCA visualization.

    renderSearchRanking();

    updateVectorPlot();

    // Finish when all selected comparisons are displayed.

    if (vectorSearch.position >= vectorSearch.order.length) {

        finishVectorSearch();

    } else {

        updateSearchControls();
    }
}


// ============================================================
// LIVE RANKING
// ============================================================

function renderSearchRanking() {

    const container = $("search-ranking");

    container.replaceChildren();

    if (!vectorSearch.examined.length) {
        return;
    }

    // Sort only the comparisons revealed so far.
    //
    // This makes the ranking evolve during playback.

    const ranked = [...vectorSearch.examined].sort(
        (a, b) =>
            b.score - a.score ||
            a.index - b.index
    );

    const topK = vectorSearch.data.topK;

    // Keep the ranking compact in the PCA tab.
    //
    // Students can inspect the complete final top-k
    // results in the Retrieval Pipeline column.

    ranked.slice(0, Math.max(5, topK)).forEach(
        (item, index) => {

            const row = document.createElement("button");

            row.type = "button";

            row.className =
                "search-ranking-item" +
                (index < topK ? " top-k" : "");

            row.style.width = "100%";
            row.style.cursor = "pointer";
            row.style.textAlign = "left";

            const rank = document.createElement("strong");

            rank.textContent = `#${index + 1}`;

            rank.style.color =
                index < topK
                    ? "#4adeb6"
                    : "#a7b9d3";

            const label = document.createElement("span");

            label.textContent =
                `Chunk ${item.index + 1}`;

            const score = document.createElement("span");

            score.className = "score";

            score.textContent = item.score.toFixed(4);

            row.append(rank, label, score);

            row.title = "Click to inspect this chunk";

            row.addEventListener("click", () => {
                inspectChunkFromPlot(item.index);
            });

            container.append(row);
        }
    );
}


// ============================================================
// UPDATE PCA ANIMATION
// ============================================================

function updateVectorPlot() {

    if (
        !window.Plotly ||
        !$("plot").data ||
        !vectorSearch.data
    ) {
        return;
    }

    const query = vectorSearch.data.queryPoint;

    const examinedPoints = vectorSearch.examined
        .map(item => documentPoints[item.index])
        .filter(Boolean);

    const currentIndex = vectorSearch.currentChunk;

    const currentPoint = currentIndex !== null
        ? documentPoints[currentIndex]
        : null;

    const finished =
        vectorSearch.position >= vectorSearch.order.length &&
        vectorSearch.order.length > 0;

    // --------------------------------------------------------
    // 1. EXAMINED CHUNKS
    // --------------------------------------------------------

    const examinedX = examinedPoints.map(p => p[0]);
    const examinedY = examinedPoints.map(p => p[1]);

    // --------------------------------------------------------
    // 2. FINAL TOP-K
    // --------------------------------------------------------

    const topResults = finished
        ? vectorSearch.data.results
        : [];

    const topPoints = topResults
        .map(item => documentPoints[item.index])
        .filter(Boolean);

    // --------------------------------------------------------
    // 3. CURRENT CONNECTION
    // --------------------------------------------------------

    const connectionX = currentPoint
        ? [query[0], currentPoint[0]]
        : [];

    const connectionY = currentPoint
        ? [query[1], currentPoint[1]]
        : [];

    // --------------------------------------------------------
    // 4. UPDATE ALL ANIMATION TRACES
    // --------------------------------------------------------

    // Plotly.restyle updates existing traces without
    // reconstructing the entire plot.

    Plotly.restyle(
        $("plot"),
        {
            x: [examinedX],
            y: [examinedY]
        },
        [1]
    );

    Plotly.restyle(
        $("plot"),
        {
            x: [topPoints.map(p => p[0])],
            y: [topPoints.map(p => p[1])],

            text: [
                topResults.map(
                    (item, index) =>
                        `#${index + 1} · Chunk ${item.index + 1}`
                )
            ],

            customdata: [
                topResults.map(item => item.index)
            ]
        },
        [2]
    );

    Plotly.restyle(
        $("plot"),
        {
            x: [connectionX],
            y: [connectionY]
        },
        [3]
    );

    Plotly.restyle(
        $("plot"),
        {
            x: [currentPoint ? [currentPoint[0]] : []],
            y: [currentPoint ? [currentPoint[1]] : []],

            text: [
                currentPoint
                    ? [`Chunk ${currentIndex + 1}`]
                    : []
            ]
        },
        [4]
    );

    Plotly.restyle(
        $("plot"),
        {
            x: [[query[0]]],
            y: [[query[1]]]
        },
        [5]
    );
}


// ============================================================
// FINISH SEARCH
// ============================================================

function finishVectorSearch() {

    clearSearchTimer();

    vectorSearch.playing = false;

    if (!vectorSearch.data) {
        return;
    }

    $("vector-search-state").textContent = "Complete";

    const total = vectorSearch.data.similarities.length;
    const animated = vectorSearch.order.length;

    $("search-progress-text").textContent =
        animated === total
            ? `All ${total} chunks compared. Top-k selected.`
            : `${animated} comparisons replayed · ` +
            `${total} chunks evaluated by the backend.`;

    $("search-progress-count").textContent =
        `${animated} / ${animated}`;

    $("search-progress").value = animated;

    // Show the actual final top-k results.
    //
    // The final ranking comes from the complete backend
    // search, including comparisons omitted from playback.

    renderFinalSearchRanking();

    updateVectorPlot();

    updateSearchControls();
}


// ============================================================
// FINAL RANKING
// ============================================================

function renderFinalSearchRanking() {

    if (!vectorSearch.data) {
        return;
    }

    const container = $("search-ranking");

    container.replaceChildren();

    const results = vectorSearch.data.results;

    results.forEach((item, index) => {

        const row = document.createElement("button");

        row.type = "button";

        row.className = "search-ranking-item top-k";

        row.style.width = "100%";
        row.style.cursor = "pointer";
        row.style.textAlign = "left";

        const rank = document.createElement("strong");

        rank.textContent = `#${index + 1}`;

        rank.style.color = "#4adeb6";

        const label = document.createElement("span");

        label.textContent = `Chunk ${item.index + 1}`;

        const score = document.createElement("span");

        score.className = "score";

        score.textContent = Number(item.score).toFixed(4);

        row.append(rank, label, score);

        row.addEventListener("click", () => {

            // Open the corresponding retrieved passage
            // in the middle column.

            renderRetrievalResults(results);

            selectRetrievedPassage(index);
        });

        container.append(row);
    });

    $("search-ranking-caption").textContent =
        `Final top-${results.length} · ` +
        `${vectorSearch.data.similarities.length} chunks searched`;
}


// ============================================================
// PLAY SEARCH
// ============================================================

function playVectorSearch() {

    if (!vectorSearch.data) {
        return;
    }

    if (vectorSearch.playing) {
        return;
    }

    if (vectorSearch.position >= vectorSearch.order.length) {
        return;
    }

    vectorSearch.playing = true;

    vectorSearch.generation += 1;

    const generation = vectorSearch.generation;

    updateSearchControls();

    $("vector-search-state").textContent = "Playing";

    const tick = () => {

        if (
            !vectorSearch.playing ||
            generation !== vectorSearch.generation
        ) {
            return;
        }

        advanceVectorSearch();

        if (
            vectorSearch.playing &&
            vectorSearch.position < vectorSearch.order.length
        ) {

            vectorSearch.timer = setTimeout(
                tick,
                searchSpeedDelay()
            );
        }
    };

    // Start immediately.
    tick();
}


// ============================================================
// NEXT STEP
// ============================================================

function nextVectorSearch() {

    if (!vectorSearch.data) {
        return;
    }

    if (vectorSearch.playing) {
        pauseVectorSearch();
    }

    advanceVectorSearch();
}


// ============================================================
// REPLAY SEARCH
// ============================================================

function replayVectorSearch() {

    if (!vectorSearch.data) {
        return;
    }

    pauseVectorSearch();

    vectorSearch.position = 0;
    vectorSearch.examined = [];
    vectorSearch.currentChunk = null;

    $("search-progress").value = 0;

    $("search-progress-count").textContent =
        `0 / ${vectorSearch.order.length}`;

    $("search-progress-text").textContent =
        "Ready to replay vector search.";

    $("vector-search-state").textContent = "Ready";

    $("search-comparison").hidden = true;

    $("search-ranking").replaceChildren();

    $("search-ranking-caption").textContent =
        "Highest-scoring chunks examined so far";

    updateVectorPlot();

    updateSearchControls();

    playVectorSearch();
}


// ============================================================
// ANIMATION BUTTON EVENTS
// ============================================================

$("search-play").addEventListener(
    "click",
    playVectorSearch
);

$("search-pause").addEventListener(
    "click",
    pauseVectorSearch
);

$("search-next").addEventListener(
    "click",
    nextVectorSearch
);

$("search-replay").addEventListener(
    "click",
    replayVectorSearch
);


// ============================================================
// PASS 5 — PCA VS. ACTUAL SIMILARITY
// ============================================================


// ============================================================
// RESET COMPARISON
// ============================================================

function resetRankingComparison() {

    rankingComparison.data = null;
    rankingComparison.mode = "pca_euclidean";
    rankingComparison.view = "both";
    rankingComparison.selected = null;

    $("ranking-comparison-state").textContent =
        "Waiting for a question";

    $("comparison-mode").value = "pca_euclidean";
    $("comparison-mode").disabled = true;

    $("comparison-k").textContent = "—";
    $("comparison-shared").textContent = "—";
    $("comparison-overlap").textContent = "—";

    $("comparison-original-ranking").replaceChildren();
    $("comparison-projected-ranking").replaceChildren();

    $("comparison-projected-title").textContent =
        "2D Euclidean";

    $("comparison-projected-description").textContent =
        "Ranked by Euclidean distance in the 2D PCA projection.";

    $("comparison-interpretation").hidden = true;

    $("comparison-overlap")
        .closest(".comparison-overlap-stat")
        ?.classList.remove(
            "high-overlap",
            "low-overlap"
        );

    for (const view of [
        "original",
        "projected",
        "both"
    ]) {

        const button = $(
            `comparison-show-${view}`
        );

        button.disabled = true;
        button.classList.remove("active");
    }

    updateComparisonPlot();
}


// ============================================================
// PREPARE COMPARISON DATA
// ============================================================

function prepareRankingComparison(data) {

    resetRankingComparison();

    const comparison = data.comparison;

    if (!comparison || !comparison.available) {

        $("ranking-comparison-state").textContent =
            comparison?.reason ||
            "Comparison unavailable";

        return;
    }

    const required = [
        comparison.original_cosine,
        comparison.pca_euclidean,
        comparison.pca_cosine
    ];

    if (
        required.some(
            item => !Array.isArray(item?.ranking)
        )
    ) {

        console.warn(
            "Invalid ranking comparison data."
        );

        return;
    }

    rankingComparison.data = comparison;

    rankingComparison.mode = "pca_euclidean";
    rankingComparison.view = "both";
    rankingComparison.selected = null;

    $("comparison-mode").disabled = false;

    for (const view of [
        "original",
        "projected",
        "both"
    ]) {

        $(`comparison-show-${view}`).disabled = false;
    }

    renderRankingComparison();
}


// ============================================================
// GET ACTIVE RANKINGS
// ============================================================

function getComparisonRankings() {

    const comparison = rankingComparison.data;

    if (!comparison) {
        return null;
    }

    const k = comparison.k;

    const original = comparison.original_cosine
        .ranking
        .slice(0, k);

    const projected = comparison[
        rankingComparison.mode
    ].ranking.slice(0, k);

    const originalSet = new Set(
        original.map(item => item.index)
    );

    const projectedSet = new Set(
        projected.map(item => item.index)
    );

    const shared = [...originalSet].filter(
        index => projectedSet.has(index)
    ).length;

    return {
        k,
        original,
        projected,
        originalSet,
        projectedSet,
        shared,
        overlap: shared / k
    };
}


// ============================================================
// RENDER COMPARISON
// ============================================================

function renderRankingComparison() {

    const comparison = rankingComparison.data;

    if (!comparison) {
        return;
    }

    const rankings = getComparisonRankings();

    if (!rankings) {
        return;
    }

    const {
        k,
        original,
        projected,
        originalSet,
        projectedSet,
        shared,
        overlap
    } = rankings;

    // --------------------------------------------------------
    // UPDATE STATUS
    // --------------------------------------------------------

    $("ranking-comparison-state").textContent =
        "Ready";

    $("comparison-k").textContent =
        String(k);

    $("comparison-shared").textContent =
        `${shared} / ${k}`;

    $("comparison-overlap").textContent =
        `${(overlap * 100).toFixed(1)}%`;

    // --------------------------------------------------------
    // OVERLAP COLORS
    // --------------------------------------------------------

    const overlapCard = $("comparison-overlap")
        .closest(".comparison-overlap-stat");

    if (overlapCard) {

        overlapCard.classList.toggle(
            "high-overlap",
            overlap === 1
        );

        overlapCard.classList.toggle(
            "low-overlap",
            overlap < 1
        );
    }

    // --------------------------------------------------------
    // PROJECTED RANKING DESCRIPTION
    // --------------------------------------------------------

    const usingEuclidean =
        rankingComparison.mode === "pca_euclidean";

    $("comparison-projected-title").textContent =
        usingEuclidean
            ? "2D Euclidean"
            : "2D Cosine";

    $("comparison-projected-description").textContent =
        usingEuclidean
            ? "Ranked by Euclidean distance in the 2D PCA projection."
            : "Ranked by cosine similarity between projected 2D vectors.";

    // --------------------------------------------------------
    // RENDER RANKING LISTS
    // --------------------------------------------------------

    renderComparisonRankingList(
        "comparison-original-ranking",
        original,
        projectedSet,
        "original"
    );

    renderComparisonRankingList(
        "comparison-projected-ranking",
        projected,
        originalSet,
        "projected"
    );

    // --------------------------------------------------------
    // INTERPRETATION
    // --------------------------------------------------------

    let interpretation;

    if (overlap === 1) {

        interpretation =
            `All ${k} nearest neighbors are shared ` +
            "between the two rankings. " +
            "However, this does not mean PCA always " +
            "preserves nearest neighbors.";

    } else {

        interpretation =
            `Only ${shared} out of ${k} nearest ` +
            "neighbors are shared between the two " +
            `rankings (Overlap@${k} = ` +
            `${(overlap * 100).toFixed(1)}%). `;
    }

    if (usingEuclidean) {

        interpretation +=
            " This comparison changes both the " +
            "representation (384D to 2D) and the " +
            "similarity metric (cosine to Euclidean).";

    } else {

        interpretation +=
            " Both rankings use cosine similarity, " +
            "but PCA reduces dimensionality and " +
            "centers the embeddings, which can " +
            "change their cosine similarities.";
    }

    interpretation +=
        " Actual RAG retrieval always uses " +
        "the original embedding space.";

    $("comparison-interpretation-text").textContent =
        interpretation;

    $("comparison-interpretation").hidden = false;

    // --------------------------------------------------------
    // UPDATE BUTTON STATES
    // --------------------------------------------------------

    const buttons = {
        original: $("comparison-show-original"),
        projected: $("comparison-show-projected"),
        both: $("comparison-show-both")
    };

    for (const [view, button] of Object.entries(buttons)) {

        button.classList.toggle(
            "active",
            rankingComparison.view === view
        );
    }

    updateComparisonPlot();
}


// ============================================================
// RENDER A RANKING LIST
// ============================================================

function renderComparisonRankingList(
    containerId,
    items,
    otherSet,
    type
) {

    const container = $(containerId);

    container.replaceChildren();

    items.forEach((item, rank) => {

        const index = Number(item.index);

        const shared = otherSet.has(index);

        const row = document.createElement("button");

        row.type = "button";

        row.className = "comparison-ranking-item";

        if (shared) {

            row.classList.add("shared");

        } else if (type === "original") {

            row.classList.add("original-only");

        } else {

            row.classList.add("projected-only");
        }

        row.classList.toggle(
            "selected",
            rankingComparison.selected === index
        );

        // ----------------------------------------------------
        // HEADER
        // ----------------------------------------------------

        const header = add(
            row,
            "comparison-ranking-item-header"
        );

        add(
            header,
            "comparison-ranking-item-rank",
            `#${rank + 1}`
        );

        let scoreLabel;

        if (type === "original") {

            scoreLabel =
                `cos ${Number(item.score).toFixed(4)}`;

        } else if (
            rankingComparison.mode === "pca_euclidean"
        ) {

            scoreLabel =
                `dist ${Number(item.distance).toFixed(4)}`;

        } else {

            scoreLabel =
                `cos ${Number(item.score).toFixed(4)}`;
        }

        add(
            header,
            "comparison-ranking-item-score",
            scoreLabel
        );

        // ----------------------------------------------------
        // CHUNK NAME
        // ----------------------------------------------------

        add(
            row,
            "comparison-ranking-item-name",
            `Chunk ${index + 1}` +
            (shared ? " · Shared" : "")
        );

        // ----------------------------------------------------
        // TEXT PREVIEW
        // ----------------------------------------------------

        const chunk = currentChunks.find(
            item => Number(item.index) === index
        );

        add(
            row,
            "comparison-ranking-item-preview",
            chunk?.text || ""
        );

        // ----------------------------------------------------
        // CLICK TO INSPECT
        // ----------------------------------------------------

        row.addEventListener("click", () => {

            rankingComparison.selected = index;

            renderRankingComparison();

            inspectChunkFromPlot(index);
        });

        container.append(row);
    });
}


// ============================================================
// UPDATE PCA HIGHLIGHTS
// ============================================================

function updateComparisonPlot() {

    if (
        !window.Plotly ||
        !$("plot").data ||
        $("plot").data.length < 9
    ) {
        return;
    }

    const rankings = getComparisonRankings();

    const groups = [
        [],
        [],
        []
    ];

    if (rankings) {

        const {
            originalSet,
            projectedSet
        } = rankings;

        const allIndices = new Set([
            ...originalSet,
            ...projectedSet
        ]);

        for (const index of allIndices) {

            const inOriginal = originalSet.has(index);
            const inProjected = projectedSet.has(index);

            const shared = inOriginal && inProjected;

            let group;

            if (shared) {

                group = 2;

            } else if (inOriginal) {

                group = 0;

            } else {

                group = 1;
            }

            const view = rankingComparison.view;

            const visible =
                view === "both" ||
                (view === "original" && inOriginal) ||
                (view === "projected" && inProjected);

            if (visible) {
                groups[group].push(index);
            }
        }
    }

    // --------------------------------------------------------
    // UPDATE TRACES 6, 7, 8
    // --------------------------------------------------------

    groups.forEach((indices, group) => {

        const points = indices
            .map(index => ({
                index,
                point: documentPoints[index]
            }))
            .filter(item => Array.isArray(item.point));

        Plotly.restyle(
            $("plot"),
            {
                x: [
                    points.map(item => item.point[0])
                ],

                y: [
                    points.map(item => item.point[1])
                ],

                customdata: [
                    points.map(item => item.index)
                ],

                text: [
                    points.map(
                        item => `Chunk ${item.index + 1}`
                    )
                ]
            },
            [6 + group]
        );
    });
}


// ============================================================
// COMPARISON CONTROLS
// ============================================================

$("comparison-mode").addEventListener(
    "change",
    () => {

        rankingComparison.mode =
            $("comparison-mode").value;

        renderRankingComparison();
    }
);


$("comparison-show-original").addEventListener(
    "click",
    () => {

        rankingComparison.view = "original";

        renderRankingComparison();
    }
);


$("comparison-show-projected").addEventListener(
    "click",
    () => {

        rankingComparison.view = "projected";

        renderRankingComparison();
    }
);


$("comparison-show-both").addEventListener(
    "click",
    () => {

        rankingComparison.view = "both";

        renderRankingComparison();
    }
);

// ============================================================
// PASS 6 — ANIMATED LLM TOKEN GENERATION
// ============================================================

const tokenGeneration = {
    tokens: [],
    position: 0,
    selected: null,
    playing: false,
    timer: null,
    temperature: 1,
    originalTemperature: 1
};


// ============================================================
// TOKEN GENERATION MODAL
// ============================================================

let tokenModalPreviousFocus = null;

function openTokenModal() {
    const modal = $("token-modal");
    tokenModalPreviousFocus = document.activeElement;
    modal.hidden = false;
    document.body.classList.add("token-modal-open");
    $("close-token-modal").focus();
}

function closeTokenModal() {
    const modal = $("token-modal");
    if (modal.hidden) return;
    pauseTokenGeneration();
    modal.hidden = true;
    document.body.classList.remove("token-modal-open");
    if (tokenModalPreviousFocus?.isConnected) {
        tokenModalPreviousFocus.focus();
    }
}

$("open-token-modal").addEventListener("click", openTokenModal);
$("close-token-modal").addEventListener("click", closeTokenModal);
$("token-modal-backdrop").addEventListener("click", closeTokenModal);

document.addEventListener("keydown", event => {
    if ($("token-modal").hidden) return;
    if (event.key === "Escape") {
        event.preventDefault();
        closeTokenModal();
    } else if (event.key === "Tab") {
        const dialog = document.querySelector(".token-modal-dialog");
        const controls = [...dialog.querySelectorAll(
            'button:not([disabled]), select:not([disabled]), input:not([disabled]), [tabindex="0"]'
        )].filter(element => element.getClientRects().length);
        if (!controls.length) return;
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }
});

function updateChatSettingsSummary() {
    $("chat-settings-summary").textContent =
        `Top-k: ${$("topk").value} · ` +
        `Temp: ${$("temperature").value} · ` +
        `History: ${$("history-length").value}`;
}

for (const id of ["topk", "temperature", "history-length"]) {
    $(id).addEventListener("input", updateChatSettingsSummary);
    $(id).addEventListener("change", updateChatSettingsSummary);
}

// ============================================================
// TOKEN ANIMATION UTILITIES
// ============================================================

function tokenSpeedDelay() {

    const speed = $("token-speed").value;

    if (speed === "slow") return 900;
    if (speed === "fast") return 120;

    return 350;
}


function stopTokenTimer() {

    if (tokenGeneration.timer !== null) {
        clearTimeout(tokenGeneration.timer);
        tokenGeneration.timer = null;
    }
}


function updateTokenControls() {

    const available = tokenGeneration.tokens.length > 0;

    const finished =
        tokenGeneration.position >= tokenGeneration.tokens.length;

    $("token-play").disabled =
        !available || tokenGeneration.playing || finished;

    $("token-pause").disabled =
        !available || !tokenGeneration.playing;

    $("token-next").disabled =
        !available || tokenGeneration.playing || finished;

    $("token-replay").disabled = !available;
    $("token-speed").disabled = !available;

    $("token-temperature-slider").disabled = !available;
}


// ============================================================
// RESET TOKEN GENERATION
// ============================================================

function resetTokenGeneration() {

    stopTokenTimer();

    tokenGeneration.tokens = [];
    tokenGeneration.position = 0;
    tokenGeneration.selected = null;
    tokenGeneration.playing = false;
    tokenGeneration.temperature = 1;
    tokenGeneration.originalTemperature = 1;

    $("token-generation-state").textContent =
        "Waiting for an answer";

    $("token-total").textContent = "—";
    $("token-current-step").textContent = "—";
    $("token-original-temperature").textContent = "—";

    $("token-progress-text").textContent =
        "No generation to replay yet.";

    $("token-progress-count").textContent = "0 / 0";

    $("token-progress").max = 1;
    $("token-progress").value = 0;

    $("token-output").replaceChildren();

    add(
        $("token-output"),
        "hint",
        "Generate an answer to inspect its tokens."
    );

    $("token-probabilities").replaceChildren();

    add(
        $("token-probabilities"),
        "hint",
        "Select a generation step to inspect its candidate tokens."
    );

    $("token-inspector").hidden = true;

    $("token-temperature-slider").value = "1";
    $("token-temperature-value").textContent = "1.0";

    updateTokenControls();
}


// ============================================================
// PREPARE RECORDED GENERATION
// ============================================================

function prepareTokenGeneration(generation) {

    resetTokenGeneration();

    if (!Array.isArray(generation?.tokens) ||
        generation.tokens.length === 0) {

        $("token-generation-state").textContent =
            "Token trace unavailable";

        return;
    }

    tokenGeneration.tokens = generation.tokens;
    tokenGeneration.originalTemperature =
        Number(generation.temperature ?? 1);

    $("token-total").textContent =
        String(tokenGeneration.tokens.length);

    $("token-original-temperature").textContent =
        tokenGeneration.originalTemperature.toFixed(1);

    $("token-generation-state").textContent = "Ready";

    $("token-progress").max =
        tokenGeneration.tokens.length;

    $("token-progress-text").textContent =
        "Ready to replay the generated answer.";

    $("token-progress-count").textContent =
        `0 / ${tokenGeneration.tokens.length}`;

    updateTokenControls();
}


// ============================================================
// FORMAT TOKEN TEXT
// ============================================================

function readableToken(token) {

    const text = String(token ?? "");

    if (text === "\n") return "↵";
    if (text === "\t") return "⇥";
    if (text === " ") return "␣";

    return text
        .replaceAll("\n", "↵")
        .replaceAll("\t", "⇥")
        .replaceAll(" ", "␣");
}


// ============================================================
// TEMPERATURE-ADJUSTED CANDIDATES
// ============================================================

// The backend returns log probabilities for a subset of tokens.
//
// For temperature exploration, we rescale the recorded
// candidate log probabilities and normalize WITHIN THAT SET.
//
// These are conditional probabilities among the displayed
// candidates, NOT probabilities over the full vocabulary.

function adjustedTokenCandidates(step) {

    const candidates = Array.isArray(step?.candidates)
        ? step.candidates
        : [];

    const valid = candidates.filter(candidate =>
        Number.isFinite(Number(candidate.logprob)) &&
        candidate.logprob !== null
    );

    if (!valid.length) return [];

    const temperature = Math.max(
        0.1,
        tokenGeneration.temperature
    );

    const logits = valid.map(
        candidate => Number(candidate.logprob) / temperature
    );

    const maximum = Math.max(...logits);

    const weights = logits.map(
        value => Math.exp(value - maximum)
    );

    const total = weights.reduce(
        (sum, value) => sum + value,
        0
    );

    return valid.map((candidate, index) => ({
        ...candidate,
        adjustedProbability: weights[index] / total
    }));
}


// ============================================================
// RENDER GENERATED SEQUENCE
// ============================================================

function renderGeneratedTokens() {

    const container = $("token-output");

    container.replaceChildren();

    if (tokenGeneration.position === 0) {

        add(
            container,
            "hint",
            "Press Play or Next to reveal the generated tokens."
        );

        return;
    }

    tokenGeneration.tokens
        .slice(0, tokenGeneration.position)
        .forEach((step, index) => {

            const span = document.createElement("span");

            span.className = "generated-token";

            if (index === tokenGeneration.position - 1) {
                span.classList.add("current");
            }

            if (index === tokenGeneration.selected) {
                span.classList.add("selected");
            }

            span.textContent = step.token ?? "";

            span.title =
                `Token ${index + 1}: ${readableToken(step.token)}`;

            span.addEventListener("click", () => {
                tokenGeneration.selected = index;
                renderGeneratedTokens();
                renderTokenStep(index);
            });

            container.append(span);
        });

    container.scrollTop = container.scrollHeight;
}


// ============================================================
// RENDER NEXT-TOKEN CANDIDATES
// ============================================================

function renderTokenCandidates(step) {

    const container = $("token-probabilities");

    container.replaceChildren();

    const candidates = adjustedTokenCandidates(step);

    if (!candidates.length) {

        add(
            container,
            "hint",
            "Candidate probabilities were not returned by the model."
        );

        return;
    }

    const selectedToken = String(step.token ?? "");

    candidates.forEach(candidate => {

        const probability = candidate.adjustedProbability;

        const chosen =
            String(candidate.token ?? "") === selectedToken;

        const row = add(
            container,
            `token-candidate${chosen ? " chosen" : ""}`
        );

        const header = add(
            row,
            "token-candidate-header"
        );

        add(
            header,
            "token-candidate-text",
            readableToken(candidate.token)
        );

        add(
            header,
            "token-candidate-probability",
            `${(probability * 100).toFixed(1)}%`
        );

        const track = add(
            row,
            "token-candidate-track"
        );

        const fill = add(
            track,
            "token-candidate-fill"
        );

        fill.style.width =
            `${Math.max(0, Math.min(100, probability * 100))}%`;
    });
}


// ============================================================
// RENDER SELECTED TOKEN
// ============================================================

function renderTokenStep(index) {

    const step = tokenGeneration.tokens[index];

    if (!step) return;

    $("token-current-step").textContent =
        String(index + 1);

    renderTokenCandidates(step);

    $("token-inspector").hidden = false;

    $("token-selected-text").textContent =
        readableToken(step.token);

    const probability = Number(step.probability);

    $("token-selected-probability").textContent =
        step.probability !== null &&
            step.probability !== undefined &&
            Number.isFinite(probability)
            ? `${(probability * 100).toFixed(2)}%`
            : "Unavailable";
}


// ============================================================
// ADVANCE ONE TOKEN
// ============================================================

function advanceTokenGeneration() {

    if (tokenGeneration.position >= tokenGeneration.tokens.length) {
        finishTokenGeneration();
        return;
    }

    tokenGeneration.position += 1;

    const index = tokenGeneration.position - 1;

    tokenGeneration.selected = index;

    renderGeneratedTokens();
    renderTokenStep(index);

    const current = tokenGeneration.position;
    const total = tokenGeneration.tokens.length;

    $("token-progress").value = current;

    $("token-progress-count").textContent =
        `${current} / ${total}`;

    $("token-progress-text").textContent =
        `Generated token ${current} of ${total}`;

    $("token-generation-state").textContent = "Generating";

    if (current >= total) {
        finishTokenGeneration();
    } else {
        updateTokenControls();
    }
}


// ============================================================
// FINISH ANIMATION
// ============================================================

function finishTokenGeneration() {

    stopTokenTimer();

    tokenGeneration.playing = false;

    $("token-generation-state").textContent = "Complete";

    $("token-progress-text").textContent =
        "All recorded tokens have been revealed.";

    updateTokenControls();
}


// ============================================================
// PLAY / PAUSE / NEXT / REPLAY
// ============================================================

function playTokenGeneration() {

    if (!tokenGeneration.tokens.length ||
        tokenGeneration.playing ||
        tokenGeneration.position >= tokenGeneration.tokens.length) {
        return;
    }

    tokenGeneration.playing = true;

    $("token-generation-state").textContent = "Playing";

    updateTokenControls();

    const tick = () => {

        if (!tokenGeneration.playing) return;

        advanceTokenGeneration();

        if (tokenGeneration.playing &&
            tokenGeneration.position < tokenGeneration.tokens.length) {

            tokenGeneration.timer = setTimeout(
                tick,
                tokenSpeedDelay()
            );
        }
    };

    tick();
}


function pauseTokenGeneration() {

    tokenGeneration.playing = false;

    stopTokenTimer();

    if (tokenGeneration.tokens.length &&
        tokenGeneration.position < tokenGeneration.tokens.length) {

        $("token-generation-state").textContent = "Paused";
    }

    updateTokenControls();
}


function nextTokenGeneration() {

    if (!tokenGeneration.tokens.length) return;

    if (tokenGeneration.playing) {
        pauseTokenGeneration();
    }

    advanceTokenGeneration();
}


function replayTokenGeneration() {

    if (!tokenGeneration.tokens.length) return;

    pauseTokenGeneration();

    tokenGeneration.position = 0;
    tokenGeneration.selected = null;

    $("token-current-step").textContent = "—";

    $("token-progress").value = 0;

    $("token-progress-count").textContent =
        `0 / ${tokenGeneration.tokens.length}`;

    $("token-progress-text").textContent =
        "Ready to replay the generated answer.";

    $("token-generation-state").textContent = "Ready";

    $("token-inspector").hidden = true;

    $("token-probabilities").replaceChildren();

    add(
        $("token-probabilities"),
        "hint",
        "Press Play or Next to inspect the first token."
    );

    renderGeneratedTokens();

    updateTokenControls();

    playTokenGeneration();
}


// ============================================================
// TOKEN GENERATION EVENT LISTENERS
// ============================================================

$("token-play").addEventListener(
    "click",
    playTokenGeneration
);

$("token-pause").addEventListener(
    "click",
    pauseTokenGeneration
);

$("token-next").addEventListener(
    "click",
    nextTokenGeneration
);

$("token-replay").addEventListener(
    "click",
    replayTokenGeneration
);

$("token-temperature-slider").addEventListener(
    "input",
    () => {

        tokenGeneration.temperature =
            Number($("token-temperature-slider").value);

        $("token-temperature-value").textContent =
            tokenGeneration.temperature.toFixed(1);

        const index = tokenGeneration.selected;

        if (index !== null) {
            renderTokenCandidates(tokenGeneration.tokens[index]);
        }
    }
);

// ============================================================
// CHAT EMPTY STATE
// ============================================================

function renderEmptyChat() {

    $("messages").replaceChildren();

    const empty = add(
        $("messages"),
        "chat-empty-state"
    );

    add(empty, "chat-empty-icon", "💬");

    const title = document.createElement("h3");

    title.textContent = "Ask your document";

    empty.append(title);

    add(
        empty,
        "hint",
        "Upload a PDF, generate its embeddings, " +
        "and ask questions about its content."
    );
}


function removeEmptyChat() {

    const empty = $("messages").querySelector(
        ".chat-empty-state"
    );

    if (empty) {
        empty.remove();
    }

    const oldHint = $("messages").querySelector(
        ":scope > .hint"
    );

    if (oldHint) {
        oldHint.remove();
    }
}


// ============================================================
// CHAT MESSAGE RENDERING
// ============================================================

function scrollChatToBottom() {

    $("messages").scrollTop =
        $("messages").scrollHeight;
}


function addChatMessage(role, text, sources = []) {

    removeEmptyChat();

    const message = add(
        $("messages"),
        `chat-message ${role}`
    );

    add(
        message,
        "chat-avatar",
        role === "user" ? "YOU" : "AI"
    );

    const bubble = add(
        message,
        "chat-bubble"
    );

    add(
        bubble,
        "chat-role",
        role === "user" ? "You" : "Assistant"
    );

    const content = add(
        bubble,
        role === "assistant"
            ? "chat-markdown"
            : "chat-text"
    );

    if (role === "assistant") {

        if (
            window.marked &&
            window.DOMPurify
        ) {

            const html = marked.parse(
                String(text || "")
            );

            content.innerHTML = DOMPurify.sanitize(html);

        } else {

            content.textContent = text;
            content.style.whiteSpace = "pre-wrap";
        }

    } else {

        content.textContent = text;
        content.style.whiteSpace = "pre-wrap";
    }

    if (
        role === "assistant" &&
        sources.length
    ) {

        const sourceContainer = add(
            bubble,
            "chat-sources"
        );

        sources.forEach((result, index) => {

            createButton(
                sourceContainer,
                "chat-source",
                `📄 Chunk ${result.index}`,
                () => {

                    renderRetrievalResults(sources);

                    selectRetrievedPassage(index);

                    document.querySelector(
                        ".retrieval-panel"
                    ).scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }
            );
        });
    }

    scrollChatToBottom();

    return message;
}


// ============================================================
// TYPING INDICATOR
// ============================================================

function showTypingIndicator(show) {

    $("typing-indicator").hidden = !show;
}


// ============================================================
// RETRIEVAL RESULTS
// ============================================================

function resetRetrieval() {

    currentResults = [];
    selectedResultIndex = null;

    $("results").replaceChildren();

    $("retrieval-count").textContent =
        "0 retrieved";

    const empty = add(
        $("results"),
        "retrieval-empty"
    );

    add(empty, "empty-icon", "🔎");

    add(
        empty,
        "",
        "No passages retrieved yet."
    );

    add(
        empty,
        "hint",
        "Ask a question to start semantic retrieval."
    );

    $("retrieval-detail").replaceChildren();
    $("retrieval-detail").hidden = true;

    $("prompt").textContent =
        "Waiting for retrieval…";

    $("prompt-details").open = false;
}


// ============================================================
// SIMILARITY BAR
// ============================================================

function createSimilarityBar(parent, score) {

    const track = add(
        parent,
        "similarity-track"
    );

    const fill = add(
        track,
        "similarity-fill"
    );

    // Cosine similarity lies in [-1, 1].
    // Map it to [0%, 100%] for visualization.
    // This is NOT a probability.

    const normalized = Math.max(
        0,
        Math.min(100, ((score + 1) / 2) * 100)
    );

    fill.style.width = `${normalized}%`;
}


// ============================================================
// RENDER RETRIEVED PASSAGES
// ============================================================

function renderRetrievalResults(results) {

    currentResults = results.map(result => ({
        ...result
    }));

    selectedResultIndex = null;

    $("results").replaceChildren();

    $("retrieval-detail").replaceChildren();
    $("retrieval-detail").hidden = true;

    $("retrieval-count").textContent =
        `${currentResults.length} retrieved`;

    if (!currentResults.length) {

        add(
            $("results"),
            "hint",
            "No relevant passages were returned."
        );

        return;
    }

    currentResults.forEach((result, index) => {

        const score = Number(result.score);

        const card = document.createElement("button");

        card.type = "button";

        card.className =
            "retrieval-card" +
            (index === 0 ? " top-result" : "");

        card.dataset.resultIndex = String(index);

        const header = add(
            card,
            "retrieval-card-header"
        );

        add(
            header,
            "retrieval-rank",
            `#${index + 1} · Chunk ${result.index}`
        );

        add(
            header,
            "retrieval-score",
            Number.isFinite(score)
                ? score.toFixed(3)
                : "N/A"
        );

        if (Number.isFinite(score)) {
            createSimilarityBar(card, score);
        }

        add(
            card,
            "retrieval-preview",
            result.text
        );

        card.addEventListener("click", () => {
            selectRetrievedPassage(index);
        });

        $("results").append(card);
    });
}


// ============================================================
// INSPECT A RETRIEVED PASSAGE
// ============================================================

function selectRetrievedPassage(index) {

    const result = currentResults[index];

    if (!result) {
        return;
    }

    selectedResultIndex = index;

    $("results")
        .querySelectorAll(".retrieval-card")
        .forEach((card, cardIndex) => {

            card.classList.toggle(
                "selected",
                cardIndex === index
            );
        });

    const detail = $("retrieval-detail");

    detail.replaceChildren();
    detail.hidden = false;

    add(
        detail,
        "retrieval-detail-title",
        `Retrieved #${index + 1} · ` +
        `Chunk ${result.index} · ` +
        `Cosine similarity: ${Number(result.score).toFixed(3)}`
    );

    add(
        detail,
        "retrieval-detail-text",
        result.text
    );

    detail.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


// ============================================================
// PROMPT INSPECTOR
// ============================================================

function updatePrompt(data) {

    $("prompt").textContent =
        `RETRIEVAL QUERY:\n` +
        `${data.search_query}\n\n` +

        `PREVIOUS EXCHANGES USED: ` +
        `${data.history_used}\n\n` +

        `RAG PROMPT:\n` +
        `${data.prompt}`;
}


$("copy-prompt").addEventListener(
    "click",
    run(async () => {

        await navigator.clipboard.writeText(
            $("prompt").textContent
        );

        const button = $("copy-prompt");

        button.textContent = "Copied!";

        setTimeout(() => {
            button.textContent = "Copy prompt";
        }, 1500);
    })
);


// ============================================================
// GENERATION STATISTICS
// ============================================================

function updateGenerationStats(
    duration,
    historyUsed
) {

    $("generation-stats").hidden = false;

    $("generation-time").textContent =
        `⏱ ${duration.toFixed(1)}s`;

    $("generation-history").textContent =
        `💬 ${historyUsed} previous exchanges`;
}


function resetGenerationStats() {

    $("generation-stats").hidden = true;

    $("generation-time").textContent = "";
    $("generation-history").textContent = "";
}


// ============================================================
// PDF EXTRACTION
// ============================================================

$("upload").onclick = run(async () => {

    if (busy) {
        return;
    }

    const file = $("pdf").files[0];

    if (!file) {
        throw Error("Choose a PDF first.");
    }

    const form = new FormData();

    form.append("file", file);

    setStatus("Extracting PDF…");

    const data = await api(
        "document",
        form,
        true
    );

    $("docstats").replaceChildren();

    const statistics = [
        ["Pages", data.pages],
        ["Words", data.words],
        ["Characters", data.characters]
    ];

    statistics.forEach(([label, value]) => {

        const box = add(
            $("docstats"),
            "stat",
            String(value)
        );

        add(box, "statlabel", label);
    });

    $("docinfo").textContent = data.preview;

    $("pages").replaceChildren();

    data.page_previews.forEach(page => {

        add(
            $("pages"),
            "page",
            `Page ${page.page} · ` +
            `${page.characters} characters\n` +
            page.preview
        );
    });

    resetChunkSearch();
    resetEmbeddingInfo();
    resetPlot();

    $("chunkstats").replaceChildren();

    await api("chat/clear", {});

    clearChatUI();

    showTab("document");

    stage(0);

    await sleep(350);

    stage(1);

    setStatus(
        "PDF extracted. Now create chunks."
    );
});


// ============================================================
// INTERACTIVE CHUNKING
// ============================================================

$("chunk").onclick = run(async () => {

    if (busy) {
        return;
    }

    setStatus("Creating chunks…");

    const data = await api("chunks", {

        chunk_size: +$("size").value,
        overlap: +$("overlap").value
    });

    currentChunks = data.chunks;

    $("chunk-search").value = "";

    $("chunk-search").disabled =
        currentChunks.length === 0;

    renderChunks();

    showTab("chunks");

    $("chunkstats").replaceChildren();

    const statistics = [
        ["Chunks", data.count],
        ["Size", data.chunk_size],
        ["Overlap", data.overlap]
    ];

    statistics.forEach(([label, value]) => {

        const box = add(
            $("chunkstats"),
            "stat",
            String(value)
        );

        add(box, "statlabel", label);
    });

    resetPlot();
    resetEmbeddingInfo();
    resetRetrieval();

    stage(2);

    setStatus(
        `${data.chunks.length} chunks created.`
    );
});


// ============================================================
// VECTORIZATION
// ============================================================

$("vectorize").onclick = run(async () => {

    if (busy) {
        return;
    }

    setStatus(
        "Loading embedding model and vectorizing " +
        "chunks (first run may take longer)…"
    );

    const data = await api(
        "vectorize",
        {}
    );

    updateEmbeddingInfo(data);

    await renderPCA(data);

    resetRetrieval();

    stage(3);

    setStatus(
        `Created ${data.points.length} embeddings ` +
        `of dimension ${data.dimension}.`
    );
});


// ============================================================
// RETRIEVAL + CHATBOT
// ============================================================

$("ask").onclick = run(async () => {

    if (busy) {
        return;
    }

    const question = $("question").value.trim();

    if (!question) {
        throw Error("Enter a question.");
    }

    const historyLength =
        +$("history-length").value;

    const topK =
        +$("topk").value;

    const temperature =
        +$("temperature").value;

    const askButton = $("ask");

    busy = true;
    askButton.disabled = true;

    try {

        // ----------------------------------------------------
        // 1. DISPLAY USER MESSAGE
        // ----------------------------------------------------

        addChatMessage(
            "user",
            question
        );

        $("question").value = "";

        // ----------------------------------------------------
        // 2. HISTORY-AWARE RETRIEVAL
        // ----------------------------------------------------

        stage(3);

        setStatus(
            "Embedding question and searching…"
        );

        await sleep(300);

        const data = await api("retrieve", {

            question,
            top_k: topK,
            history_length: historyLength
        });

        stage(4);

        await sleep(250);

        // ----------------------------------------------------
        // 3. RETRIEVED PASSAGES
        // ----------------------------------------------------

        renderRetrievalResults(
            data.results
        );

        // Snapshot for this particular answer.

        const sources = data.results.map(
            result => ({ ...result })
        );

        // ----------------------------------------------------
        // PASS 4: PREPARE ANIMATED VECTOR SEARCH
        // ----------------------------------------------------

        // This does not delay LLM generation.
        //
        // Students can replay the search whenever they
        // want using the controls in the PCA tab.

        prepareVectorSearch(data);

        stage(5);

        // ----------------------------------------------------
        // 4. PROMPT
        // ----------------------------------------------------

        updatePrompt(data);

        await sleep(250);

        // ----------------------------------------------------
        // 5. LOCAL LLM GENERATION
        // ----------------------------------------------------

        stage(6);

        setStatus(
            "Generating locally on CPU " +
            `(using ${data.history_used} ` +
            "previous exchanges)…"
        );

        showTypingIndicator(true);

        const start = performance.now();

        const output = await api("chat", {

            question,
            prompt: data.prompt,
            temperature,
            history_length: historyLength
        });

        const duration =
            (performance.now() - start) / 1000;

        // ----------------------------------------------------
        // 6. DISPLAY ANSWER
        // ----------------------------------------------------

        showTypingIndicator(false);

        addChatMessage(
            "assistant",
            output.answer,
            sources
        );

        // Pass 6: prepare the recorded token-generation animation.
        prepareTokenGeneration(output.generation);

        stage(7);

        // ----------------------------------------------------
        // 7. GENERATION STATISTICS
        // ----------------------------------------------------

        updateGenerationStats(
            duration,
            output.history_used
        );

        setStatus(
            `Done in ${duration.toFixed(1)}s · ` +
            `${output.history_used} previous exchanges used.`
        );

    } catch (error) {

        showTypingIndicator(false);

        throw error;

    } finally {

        busy = false;
        askButton.disabled = false;
    }
});


// ============================================================
// KEYBOARD SHORTCUT
// ============================================================

// Enter sends the question.
// Shift + Enter inserts a newline.

$("question").addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.isComposing
        ) {

            event.preventDefault();

            if (!busy) {
                $("ask").click();
            }
        }
    }
);


// ============================================================
// CLEAR CHAT UI
// ============================================================

function clearChatUI() {

    resetTokenGeneration();
    closeTokenModal();

    renderEmptyChat();

    $("question").value = "";

    showTypingIndicator(false);

    resetRetrieval();
    resetGenerationStats();

    resetVectorSearch();
    resetRankingComparison();

    // Restore the original document-only PCA plot.

    if (
        window.Plotly &&
        $("plot").data &&
        documentPoints.length
    ) {

        Plotly.restyle(
            $("plot"),
            {
                x: Array.from({ length: 8 }, () => []),
                y: Array.from({ length: 8 }, () => [])
            },
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    stages.forEach((element, index) => {

        if (index >= 3) {

            element.classList.remove(
                "done",
                "active"
            );
        }
    });
}


// ============================================================
// CLEAR CHAT SESSION
// ============================================================

$("clear-chat").onclick = run(async () => {

    if (busy) {
        return;
    }

    await api(
        "chat/clear",
        {}
    );

    clearChatUI();

    setStatus("Chat cleared.");
});


// ============================================================
// REMOVE PDF
// ============================================================

$("remove-pdf").onclick = run(async () => {

    if (busy) {
        return;
    }

    if (!confirm(
        "Remove the current PDF and clear " +
        "the entire session?"
    )) {
        return;
    }

    await api(
        "reset",
        {}
    );

    // Reset document.

    $("pdf").value = "";

    $("docstats").replaceChildren();

    $("docinfo").textContent =
        "Upload a text-based PDF to begin.";

    $("pages").replaceChildren();

    $("pages-details").open = false;

    // Reset chunks.

    $("chunkstats").replaceChildren();

    resetChunkSearch();

    // Reset embeddings and PCA.

    resetEmbeddingInfo();
    resetPlot();

    // Reset chat and retrieval.

    clearChatUI();

    // Reset pipeline.

    stages.forEach(element => {

        element.classList.remove(
            "done",
            "active"
        );
    });

    showTab("document");

    setStatus(
        "PDF removed. Upload a new document to begin."
    );
});


// ============================================================
// INITIALIZE APPLICATION
// ============================================================

function initialize() {

    showTab("document");

    $("token-modal").hidden = true;
    document.body.classList.remove("token-modal-open");
    updateChatSettingsSummary();
    resetTokenGeneration();

    renderEmptyChat();

    resetRetrieval();

    resetGenerationStats();

    resetEmbeddingInfo();

    resetVectorSearch();

    showTypingIndicator(false);

    resetRankingComparison();
}

initialize();