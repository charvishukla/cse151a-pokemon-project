const fs = require('fs');
OUTPUT_FILE = "images.json";
async function writeFileAsync(filename, data) {
    try {
        await fs.writeFile(filename, data, 'utf8');
        console.log('File written successfully');
    } catch (error) {
        console.error('Error writing file:', error);
    }
}

async function executeAllAsyncSettled(asyncFunctions) {
    const results = await Promise.allSettled(asyncFunctions.map(async (fn, index) => {
        console.log(`Starting promise ${index + 1}`);
        const result = await fn();
        console.log(`Finished promise ${index + 1}`);
        return result;
    }));

    const fulfilled = results.filter(result => result.status === 'fulfilled').map(result => result.value);
    const rejected = results.filter(result => result.status === 'rejected').map(result => result.reason);
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(imageMaps));
    console.log('Fulfilled results:', fulfilled.length);
    console.log('Rejected results:', rejected.length);
}
// const API_KEY = "";//Not required
// if(!API_KEY || API_KEY !== ""){
//     const myHeaders = new Headers();
//     myHeaders.append("X-Api-Key", API_KEY);
//     const requestOptions = {
//         method: "GET",
//         redirect: "follow",
//         headers: myHeaders,
//     };
// }
// else{
//     const requestOptions = {
//         method: "GET",
//         redirect: "follow",
//     };
// }
const requestOptions = {
    method: "GET",
    redirect: "follow",
};
const imageMaps = [];

async function getImages(pageNum){
    pageImages = {};
    await fetch(`https://api.pokemontcg.io/v2/cards?page=${pageNum}&pageSize=250`, requestOptions)
    .then((response) => response.text())
    .then((result) => {
        let page = JSON.parse(result);
        if(!page || !page["data"]){
            throw new Error("Error fetching page " + pageNum);
        }
        else if(page["data"].length === 0){
            throw new Error("No data found on page " + pageNum);
        }
        page["data"].forEach((card) => {
            // console.log(card.id)
            pageImages[card.id] = card.images.large || null;
        });
})
.catch((error) => console.error(error.message));
    return pageImages;
}

functionsToExecute = [];
for(let i = 1; i < 74; i++){
    functionsToExecute.push(async () => {
        imageMaps.push(await getImages(i));
    });
}

executeAllAsyncSettled(functionsToExecute)
