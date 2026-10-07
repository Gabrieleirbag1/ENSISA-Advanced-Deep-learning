/* ============================================================
   Navigation
   ============================================================ */

const navigationButtons = document.querySelectorAll(".nav-button");
const views = document.querySelectorAll(".view");

navigationButtons.forEach((button) => {
    button.addEventListener("click", () => {

        const targetView = button.dataset.view;

        navigationButtons.forEach((btn) => {
            btn.classList.remove("active");
        });

        button.classList.add("active");

        views.forEach((view) => {
            view.classList.remove("active");
        });

        document
            .getElementById(`${targetView}-view`)
            .classList.add("active");
    });
});


/* ============================================================
   Photo inspection panel
   ============================================================ */

const photoPanel = document.getElementById("photo-panel");
const closePhotoPanelButton =
    document.getElementById("close-photo-panel");

const selectedPhoto =
    document.getElementById("selected-photo");

const selectedPhotoName =
    document.getElementById("selected-photo-name");

const neighbourGrid =
    document.getElementById("neighbour-grid");


async function openPhotoPanel(photo) {

    selectedPhoto.src = photo.url;
    selectedPhoto.alt = photo.name;

    selectedPhotoName.textContent = photo.name;

    neighbourGrid.innerHTML = `
        <p>Finding nearest photos...</p>
    `;

    photoPanel.classList.remove("hidden");


    try {

        const response = await fetch(
            `/api/neighbours/${encodeURIComponent(photo.name)}`
        );

        if (!response.ok) {
            throw new Error("Could not load nearest photos.");
        }

        const neighbours = await response.json();

        renderNeighbours(neighbours);

    } catch (error) {

        console.error(error);

        neighbourGrid.innerHTML = `
            <p>Could not load nearest photos.</p>
        `;

    }

}


function renderNeighbours(neighbours) {

    neighbourGrid.innerHTML = "";

    neighbours.forEach((neighbour) => {

        const card = document.createElement("div");

        card.className = "neighbour-card";


        const image = document.createElement("img");

        image.src = neighbour.url;
        image.alt = neighbour.name;


        const similarity = document.createElement("span");

        similarity.className = "similarity";

        similarity.textContent =
            `${(neighbour.similarity * 100).toFixed(0)}%`;


        card.appendChild(image);
        card.appendChild(similarity);


        // Clicking a neighbour makes it the selected photo.
        card.addEventListener("click", () => {
            openPhotoPanel(neighbour);
        });


        neighbourGrid.appendChild(card);

    });

}


function closePhotoPanel() {
    photoPanel.classList.add("hidden");
}


closePhotoPanelButton.addEventListener("click", closePhotoPanel);


// Close when clicking outside the modal
photoPanel.addEventListener("click", (event) => {

    if (event.target === photoPanel) {
        closePhotoPanel();
    }

});


// Close using Escape
document.addEventListener("keydown", (event) => {

    if (event.key === "Escape") {
        closePhotoPanel();
    }

});


/* ============================================================
   Photo collection
   ============================================================ */

const photoGrid = document.getElementById("photo-grid");
const photoCount = document.getElementById("photo-count");


async function loadPhotos() {

    try {

        const response = await fetch("/api/photos");

        if (!response.ok) {
            throw new Error("Could not load photos.");
        }

        const photos = await response.json();

        photoCount.textContent = photos.length;

        renderPhotoCollection(photos);

    } catch (error) {

        console.error(error);

        photoGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">!</div>

                <h2>Could not load photos</h2>

                <p>${error.message}</p>
            </div>
        `;

    }
}


function renderPhotoCollection(photos) {

    photoGrid.innerHTML = "";

    if (photos.length === 0) {

        photoGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">▧</div>

                <h2>No photos found</h2>

                <p>Add photos to the collection first.</p>
            </div>
        `;

        return;
    }


    photos.forEach((photo) => {

        const card = document.createElement("div");

        card.className = "photo-card";


        const image = document.createElement("img");

        image.src = photo.url;
        image.alt = photo.name;
        image.loading = "lazy";


        card.appendChild(image);


        card.addEventListener("click", () => {
            openPhotoPanel(photo);
        });


        photoGrid.appendChild(card);

    });

}


/* ============================================================
   Photo map
   ============================================================ */

const photoMap = document.getElementById("photo-map");


/*
 * The backend always returns normalized coordinates in [0, 1].
 *
 * MAP_PADDING controls the visual margin around the extrema.
 * The data itself remains unchanged.
 */
const MAP_PADDING = 0.05;

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

let mapZoom = 1;
let mapPanX = 0;
let mapPanY = 0;

let mapDragging = false;
let mapDragStartX = 0;
let mapDragStartY = 0;
let mapPanStartX = 0;
let mapPanStartY = 0;

let mapWorld = null;


async function loadPhotoMap() {

    try {

        const response = await fetch("/api/map");

        if (!response.ok) {
            throw new Error("Could not load photo map.");
        }

        const photos = await response.json();

        renderPhotoMap(photos);

    } catch (error) {

        console.error(error);

    }

}


/*
 * Convert a normalized coordinate from [0, 1] to a padded
 * percentage inside the map.
 *
 * Example with MAP_PADDING = 0.05:
 *
 *     0.0 -> 5%
 *     0.5 -> 50%
 *     1.0 -> 95%
 */
function paddedCoordinate(value) {

    return (
        MAP_PADDING +
        value * (1 - 2 * MAP_PADDING)
    ) * 100;

}


function renderPhotoMap(photos) {

    photoMap.innerHTML = "";

    /*
     * The viewport is the existing #photo-map.
     *
     * mapWorld contains everything that moves when the user
     * zooms or pans.
     */
    mapWorld = document.createElement("div");
    mapWorld.className = "map-world";

    photoMap.appendChild(mapWorld);


    /* --------------------------------------------------------
       Axis ticks
       -------------------------------------------------------- */

    const ticks = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ];


    /*
     * X axis
     */
    ticks.forEach((value) => {

        const tick = document.createElement("div");

        tick.className = "map-axis-tick map-axis-tick-x";

        tick.style.left =
            `${paddedCoordinate(value)}%`;

        tick.textContent =
            value.toFixed(1);

        mapWorld.appendChild(tick);

    });


    /*
     * Y axis
     *
     * The browser's vertical coordinate increases downward.
     * We display the values returned by the backend directly:
     *
     *     y = 0 -> top
     *     y = 1 -> bottom
     */
    ticks.forEach((value) => {

        const tick = document.createElement("div");

        tick.className = "map-axis-tick map-axis-tick-y";

        tick.style.top =
            `${paddedCoordinate(value)}%`;

        tick.textContent =
            value.toFixed(1);

        mapWorld.appendChild(tick);

    });


    /* --------------------------------------------------------
       Grid lines
       -------------------------------------------------------- */

    ticks.forEach((value) => {

        const verticalLine =
            document.createElement("div");

        verticalLine.className =
            "map-grid-line map-grid-line-vertical";

        verticalLine.style.left =
            `${paddedCoordinate(value)}%`;

        mapWorld.appendChild(verticalLine);


        const horizontalLine =
            document.createElement("div");

        horizontalLine.className =
            "map-grid-line map-grid-line-horizontal";

        horizontalLine.style.top =
            `${paddedCoordinate(value)}%`;

        mapWorld.appendChild(horizontalLine);

    });


    /* --------------------------------------------------------
       Photos
       -------------------------------------------------------- */

    photos.forEach((photo) => {

        const image =
            document.createElement("img");

        image.className = "map-photo";

        image.src = photo.url;
        image.alt = photo.name;

        image.style.left =
            `${paddedCoordinate(photo.x)}%`;

        image.style.top =
            `${paddedCoordinate(photo.y)}%`;


        /*
         * Prevent a photo click from beginning a pan.
         */
        image.addEventListener(
            "mousedown",
            (event) => {
                event.stopPropagation();
            }
        );


        image.addEventListener("click", (event) => {

            event.stopPropagation();

            openPhotoPanel(photo);

        });


        mapWorld.appendChild(image);

    });


    resetPhotoMapView();

}


/* ============================================================
   Photo map transformations
   ============================================================ */

function updatePhotoMapTransform() {

    if (!mapWorld) {
        return;
    }

    /*
     * Scale the representation space.
     *
     * This increases the distance between points.
     */
    mapWorld.style.transform =
        `translate(${mapPanX}px, ${mapPanY}px) ` +
        `scale(${mapZoom})`;


    /*
     * Counter-scale the photographs so that zooming changes
     * their positions but not their displayed size.
     */
    const photos =
        mapWorld.querySelectorAll(".map-photo");

    photos.forEach((photo) => {

        photo.style.setProperty(
            "--map-photo-scale",
            1 / mapZoom
        );

    });

}


/*
 * Reset to the original full-map view.
 */
function resetPhotoMapView() {

    mapZoom = 1;
    mapPanX = 0;
    mapPanY = 0;

    updatePhotoMapTransform();

}


/* ============================================================
   Photo map zoom
   ============================================================ */

photoMap.addEventListener(
    "wheel",
    (event) => {

        event.preventDefault();

        if (!mapWorld) {
            return;
        }


        const rect =
            photoMap.getBoundingClientRect();


        /*
         * Cursor position relative to the viewport.
         */
        const mouseX =
            event.clientX - rect.left;

        const mouseY =
            event.clientY - rect.top;


        /*
         * Position in world coordinates before zooming.
         */
        const worldX =
            (mouseX - mapPanX) / mapZoom;

        const worldY =
            (mouseY - mapPanY) / mapZoom;


        /*
         * Smooth exponential zoom.
         */
        const zoomFactor =
            event.deltaY < 0 ? 1.15 : 1 / 1.15;

        const newZoom = Math.min(
            MAX_ZOOM,
            Math.max(
                MIN_ZOOM,
                mapZoom * zoomFactor
            )
        );


        /*
         * Keep the point underneath the cursor fixed while
         * changing the zoom.
         */
        mapPanX =
            mouseX - worldX * newZoom;

        mapPanY =
            mouseY - worldY * newZoom;

        mapZoom = newZoom;


        /*
         * Returning to zoom 1 also returns to the original
         * centered view.
         */
        if (mapZoom === MIN_ZOOM) {
            mapPanX = 0;
            mapPanY = 0;
        }


        updatePhotoMapTransform();

    },
    {
        passive: false,
    }
);


/* ============================================================
   Photo map panning
   ============================================================ */

photoMap.addEventListener(
    "mousedown",
    (event) => {

        /*
         * Only the primary mouse button pans.
         */
        if (event.button !== 0) {
            return;
        }

        mapDragging = true;

        mapDragStartX = event.clientX;
        mapDragStartY = event.clientY;

        mapPanStartX = mapPanX;
        mapPanStartY = mapPanY;

        photoMap.classList.add("dragging");

    }
);


window.addEventListener(
    "mousemove",
    (event) => {

        if (!mapDragging) {
            return;
        }

        const dx =
            event.clientX - mapDragStartX;

        const dy =
            event.clientY - mapDragStartY;

        mapPanX =
            mapPanStartX + dx;

        mapPanY =
            mapPanStartY + dy;

        updatePhotoMapTransform();

    }
);


window.addEventListener(
    "mouseup",
    () => {

        if (!mapDragging) {
            return;
        }

        mapDragging = false;

        photoMap.classList.remove("dragging");

    }
);


/*
 * Double-click anywhere on the map to return to the
 * complete representation space.
 */
photoMap.addEventListener(
    "dblclick",
    (event) => {

        event.preventDefault();

        resetPhotoMapView();

    }
);


/* ============================================================
   Groups
   ============================================================ */

const groupsGrid = document.getElementById("groups-grid");


async function loadGroups() {

    try {

        const response = await fetch("/api/groups");

        if (!response.ok) {
            throw new Error("Could not load groups.");
        }

        const groups = await response.json();

        renderGroups(groups);

    } catch (error) {

        console.error(error);

        groupsGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">!</div>

                <h2>Could not load groups</h2>

                <p>${error.message}</p>
            </div>
        `;

    }

}


function renderGroups(groups) {

    groupsGrid.innerHTML = "";

    groups.forEach((group) => {

        const card = document.createElement("div");

        card.className = "group-card";


        /* Header */

        const header = document.createElement("div");

        header.className = "group-header";


        const title = document.createElement("h2");

        title.className = "group-title";
        title.textContent = group.name;


        const count = document.createElement("span");

        count.className = "group-count";

        count.textContent =
            `${group.count} photo${group.count !== 1 ? "s" : ""}`;


        header.appendChild(title);
        header.appendChild(count);


        /* Photo preview */

        const preview = document.createElement("div");

        preview.className = "group-preview";


        group.photos.forEach((photo) => {

            const image =
                document.createElement("img");

            image.src = photo.url;
            image.alt = photo.name;
            image.loading = "lazy";

            image.addEventListener("click", () => {
                openPhotoPanel(photo);
            });

            preview.appendChild(image);

        });


        card.appendChild(header);
        card.appendChild(preview);

        groupsGrid.appendChild(card);

    });

}


/* ============================================================
   Start application
   ============================================================ */

loadPhotos();
loadPhotoMap();
loadGroups();