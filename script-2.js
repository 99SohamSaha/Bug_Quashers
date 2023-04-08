const container = document.querySelector('.container');


const myButton = document.querySelector('#run-button');

myButton.addEventListener('click', () => {

    console.log('hello')
    // TODO: Add code to be executed when the button is clicked
//     const errors = 5;
//     const errorClustering = 'Error occured'

//     //   console.log(message)

//     // Create a new message element
//     const errorElement = document.createElement('div');
//     //   errorElement.classList.add('message');
//     errorElement.innerHTML = `<div class="flex-container">
//   <div>Number of errors:</div>
//   <div>${errors}</div>
//   <div>Error Clusttering</div>
//   <div>${errorClustering}</div>
//   </div>`;

//     // Add the message element to the chat messages
//     container.appendChild(errorElement);

    //   console.log(generateResponse(message))


    //   //For response
    //   // Create a new message element
    //   const responseElement = document.createElement('div');
    //   responseElement.classList.add('message');
    //   responseElement.classList.add('response');
    //   responseElement.innerHTML = `<p>${generateResponse(message)}</p>`;

    //   // Add the message element to the chat messages
    //   chatMessages.appendChild(responseElement);

    //   // Clear the message input field
    //   messageInput.value = '';
});



//for charts
const ctx = document.getElementById('myChart');

new Chart(ctx, {
  type: 'line',
  data: {
    labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
    datasets: [{
      label: '# of Votes',
      data: [12, 19, 3, 5, 2, 3],
      
      borderWidth: 1
    }]
  },
  options: {
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
});





