const express = require("express");
const app = express();

const input_object = {
  call_id: "101",
  company_name: "BluJay Solutions",
  company_description:
    "BluJay Solutions provides transportation management and global trade network solutions.",
  customer: {
    offerings:
      "Their software optimizes logistics and customs processes, ensuring smooth and compliant cross-border operations.",
    icp: {
      target_industry: "Manufacturing, Retail, Logistics",
      employee_count: "15000",
      region: "India",
      roles: "Manager",
      min_pricing: "301847",
    },
  },
  vendor: {
    requirements: "Global Trade and Supply Chain Management Solutions",
    ivp: {
      vendor_industry: "Transportation, Logistics, Retail",
      clients_count: "100",
      region: "India",
      max_pricing: "500000",
      year_of_establishment: "3",
    },
  },
};
const input_data = JSON.stringify(input_object);

//Import PythonShell module.
const { PythonShell } = require("python-shell");

//Router to handle the incoming request.
app.get("/", async (req, res) => {
  //Here are the option object in which arguments can be passed for the python_test.js.
  let options = {
    mode: "text",
    PythonPath: "",
    pythonOptions: ["-u"], // get print results in real-time
    // scriptPath: "/main.py", //If you are having python_test.py script in same folder, then it's optional.
    // scriptPath:'C:\Users\chira\OneDrive\Desktop\python-shell\main.',
    args: [input_data], //An argument which can be accessed in the script using sys.argv[1]
  };

  PythonShell.run("main.py", options)
    .then((results) => {
      console.log(results);
    })
    .catch((e) => {
      console.log(e);
    });

  res.send("done");
});

//Creates the server on default port 8000 and can be accessed through localhost:8000
const port = 8000;
app.listen(port, () => console.log(`Server connected to ${port}`));
