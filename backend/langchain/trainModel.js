import { LangChain, LangChainDataset } from 'langchain';
import Company from '../models/company';


const langModel = new LangChain();
const companyDataset = new LangChainDataset('company');

Company.find({}, (err, companies) => {
  if (err) {
    console.error('Error fetching companies:', err);
    return;
  }
  
  companies.forEach(company => {
    companyDataset.addExample(company.name);
    companyDataset.addExample(company.profile);
    companyDataset.addExample(company.description);
    companyDataset.addExample(company.category);
  });

  // Train the LangChain model with the dataset
  langModel.train(companyDataset);
});
