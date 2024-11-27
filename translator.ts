import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

const key = "eb1ba35535d440c196b12b5a0cd86233"; // 请使用您的实际 API 密钥
const endpoint = "https://api.cognitive.microsofttranslator.com/";
const location = "westus3"; // 请使用您在 Azure 门户中看到的位置

export const translateText = async (text: string, language: string): Promise<string> => {
    try {
        let pureLang = language.toLowerCase().trim();
        const response = await axios({
            baseURL: endpoint,
            url: '/translate',
            method: 'post',
            headers: {
                'Ocp-Apim-Subscription-Key': key,
                'Ocp-Apim-Subscription-Region': location,
                'Content-type': 'application/json',
                'X-ClientTraceId': uuidv4().toString()
            },
            params: {
                'api-version': '3.0',
                'from': 'en',
                'to': pureLang
            },
            data: [{
                'text': text
            }],
            responseType: 'json'
        })

        // 假设 API 返回的结构包含翻译文本在 response.data[0].translations[0].text
        const translatedText = response.data[0].translations[0].text;
        return translatedText;

    } catch (error: any) {
        console.error('Error translating text:', error.response ? error.response.data : error.message);
        throw error;
    }
};
